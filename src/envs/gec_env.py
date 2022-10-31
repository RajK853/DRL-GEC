import os
import numpy as np
from io import StringIO
from gym import Env, spaces, utils
from gym.utils.renderer import Renderer
from gym.envs.registration import register
from typing import List, Dict, Tuple, Any, Union

from src.tokenizers import Tokenizer, WSTokenizer
from src.sampler import IndexSampler, EditMaskGenerator
from src.utils import START_TOKEN, apply_labels, load_json, load_text, filter_correct, filter_by_num_ref, \
    get_lev_dist, lev_dist_reward, adaptive_sentence_gleu


# Global variables
Tokens = List[str]
Actions = np.ndarray
Labels = np.char.array
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DEFAULT_LABELS_PATH = os.path.join(ROOT_PATH, r"data/vocabs/labels.txt")
DEFAULT_SMALL_LABELS_PATH = os.path.join(ROOT_PATH, r"data/vocabs/labels_small.txt")
MAX_EPISODE_STEPS = 5
DEFAULT_REWARD_CONFIG = {
    "correct": 1.0,
    "fn_penalty": -0.1,
    "out_of_range_reward": -1.0,
}
OUTPUT_TEXT_FMT = """\x1b[37;1mTimestep:\x1b[0m {0}  
\x1b[37;1mRewards:\x1b[0m {1}  
\x1b[37;1mSource:\x1b[0m {2}  
\x1b[37;1mOutput:\x1b[0m {3}  
"""


class GECEnvLevDist(Env):
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
    }

    def __init__(
            self,
            datasets: List[str],
            label_path: str = DEFAULT_LABELS_PATH,
            render_mode: str = "ansi",
            max_episode_steps: int = MAX_EPISODE_STEPS,
            tokenizer: Tokenizer = WSTokenizer,
            reward_config: Dict[str, float] = None,
            add_start: bool = True,
            correct_examples_percent: List[bool] = None,
            repeat_interval: int = 1_000,
            repeat: int = 3,
            consecutive: bool = False,
            min_num_refs: List[int] = None,
    ):
        self.tokenizer = tokenizer
        self.add_start = add_start
        self.max_episode_steps = max_episode_steps
        self.reward_config = DEFAULT_REWARD_CONFIG.copy()
        if reward_config:
            self.reward_config.update(reward_config)
        labels = load_text(label_path)
        self.num_actions = len(labels)
        self.labels = np.char.array(labels, unicode=True)
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Discrete(1)
        # Load and process data
        self.data = []
        if correct_examples_percent is None:
            correct_examples_percent = [1.0]*len(datasets)
        if min_num_refs is None:
            min_num_refs = [1]*len(datasets)
        for i, name in enumerate(datasets):
            data_path = os.path.join(ROOT_PATH, f"data/processed/{name}/data.json")
            data = load_json(data_path)
            print(f"Original number of data in {name}: {len(data)}")
            data = filter_correct(data, correct_examples_percent[i])
            data = filter_by_num_ref(data, min_refs=min_num_refs[i])
            print(f"Number of data without correct sentences: {len(data)}")
            self.data.extend(data)
        assert len(self.data) > 1, "No data with current setting."
        # Render configs
        self.render_mode = render_mode
        self.renderer = Renderer(self.render_mode, self._render)
        # Data configs
        self.data_i = 0
        self.num_sents = len(self.data)
        data_indexes = np.arange(self.num_sents, dtype="uint32")
        self.mask_generator = EditMaskGenerator(self.labels)
        self.index_sampler = IndexSampler(data_indexes, interval=repeat_interval, repeat=repeat, consecutive=consecutive)
        # Environment variables
        self._episode_steps = 0
        self._orig_num_tokens = 0
        self._min_num_tokens = 0
        self._max_num_tokens = 0
        self._are_tokens_correct = False
        self._token_len_in_range = True
        self._success = False
        self._prev_tokens = None
        self._prev_actions = None
        self._prev_labels = None
        self._prev_reward = None
        self._done = None
        self._max_token_num = 0
        self._current_tokens = None
        self._reference_tokens_list = None

    @property
    def reference_tokens_list(self):
        return self._reference_tokens_list

    @property
    def current_reference(self):
        ref_i = np.random.randint(0, len(self._reference_tokens_list))
        reference = self._reference_tokens_list[ref_i]
        return reference

    def init_episode_tokens(self, data_dict):
        # Tokenize source-reference pair
        self._current_tokens = self.tokenizer.text2tokens(data_dict["text"])
        self._reference_tokens_list = [self.tokenizer.text2tokens(ref) for ref in data_dict["references"]]
        if self.add_start and self._current_tokens[0] != START_TOKEN:  # Add start-token to the source-reference pair
            self._current_tokens = [START_TOKEN] + self._current_tokens
            self._reference_tokens_list = [[START_TOKEN] + ref for ref in self._reference_tokens_list]

    def reset(self, *, seed=None, return_info=False, options=None) -> Tokens:
        self.renderer.reset()
        # Select new source-reference pair
        self.data_i = self.index_sampler.sample()    # Obtain data index
        data_dict = self.data[self.data_i]           # Obtain data dict with source-reference
        self.init_episode_tokens(data_dict)
        self._max_token_num = max(1.5*len(self._current_tokens), 10)   # Maximum number of allowed tokens
        # Initialize the episode variables
        self._episode_steps = 0
        self._are_tokens_correct = False
        self._token_len_in_range = True
        self._success = False
        self._done = False
        self._orig_num_tokens = len(self._current_tokens)
        self._prev_tokens = self._current_tokens                   # No previous tokens. So set it to current tokens
        self._prev_actions = np.zeros(self._orig_num_tokens, dtype="uint32")
        self._prev_labels = self.labels[self._prev_actions]        # No previous labels. So set it to all keeps
        # Compute initial reward
        self._prev_reward = 0
        # Initialize the renderer
        self.renderer.render_step()
        return self._current_tokens

    def step(self, actions: Actions) -> Tuple[Tokens, float, bool, Dict[str, Any]]:
        assert len(self._current_tokens) == len(actions), "Number of current tokens and actions are not same!"
        labels = self.mask_generator.actions_to_labels(actions)
        new_tokens = apply_labels(self._current_tokens, labels)                                # Apply the labels
        self._are_tokens_correct = self._check_token(new_tokens, labels)
        self._token_len_in_range = self._check_token_len(new_tokens)
        self._success = (self._current_tokens == new_tokens) and self._are_tokens_correct
        reward = self.compute_reward(self._current_tokens, new_tokens, self._reference_tokens_list, labels)
        # Update the environment variables
        self._prev_tokens = self._current_tokens
        self._current_tokens = new_tokens
        self._prev_actions = actions
        self._prev_labels = labels
        self._prev_reward = reward
        self._episode_steps += 1
        self._done = self._check_done()
        info = {"success": self._success}
        self.renderer.render_step()
        return self._current_tokens, reward, self._done, info

    @staticmethod
    def _compute_reward(prev_tokens, tokens, references) -> float:
        rewards = []
        for ref_tokens in references:
            len_ref = len(ref_tokens)
            prev_ref_lev_dist = get_lev_dist(prev_tokens, ref_tokens)
            cur_ref_lev_dist = get_lev_dist(tokens, ref_tokens)
            prev_reward = lev_dist_reward(prev_ref_lev_dist, len_ref)
            current_reward = lev_dist_reward(cur_ref_lev_dist, len_ref)
            lev_reward = current_reward - prev_reward
            rewards.append(lev_reward)
        return max(rewards)

    def compute_reward(self, prev_tokens, tokens, references, labels) -> float:
        if prev_tokens == tokens:
            if self._are_tokens_correct and all(labels == "$KEEP"):
                return self.reward_config["correct"]
            else:
                return self.reward_config["fn_penalty"]
        if not self._token_len_in_range:
            return self.reward_config["out_of_range_reward"]
        reward = self._compute_reward(prev_tokens, tokens, references=references)
        return reward

    def render(self, mode: str = "ansi") -> Union[list, None, str]:
        if self.render_mode is not None:
            return self.renderer.get_renders()
        else:
            return self._render(mode)

    def _render(self, mode: str = "ansi") -> str:
        assert mode in self.metadata["render_modes"]
        if mode == "ansi":
            return self.render_text(self._prev_tokens, self._prev_labels, self._prev_reward, self._current_tokens, self._episode_steps)
        else:
            raise NotImplementedError(f"'{mode}' mode not implemented yet")

    @staticmethod
    def format_tokens(tokens: Tokens, labels: Labels) -> Tokens:
        """
        Colorize the updated tokens
        """
        output_tokens = []
        for token, label in zip(tokens, labels):
            if label == "$KEEP":
                output_tokens.append(token)
            else:
                token = utils.colorize(token, "green", bold=True)
                label = utils.colorize(label, "red", bold=True)
                output_tokens.append(f"{token} [{label}]")     # Set label to the right side of its respective token
        return output_tokens

    def render_text(self, prev_tokens, labels, reward, current_tokens, episode_step) -> str:
        with StringIO() as outfile:
            # Colorize the source tokens based on their labels
            src_tokens = self.format_tokens(prev_tokens, labels)
            # Colorize the timestep to indicate the end of the episode
            if self._done:
                timestep = utils.colorize(str(episode_step), "green", bold=True)
            else:
                timestep = episode_step
            # Convert tokens into text
            src_text = self.tokenizer.tokens2text(src_tokens)
            trg_text = self.tokenizer.tokens2text(current_tokens)
            # Obtain output to render
            rewards_text = f"{reward:.3f}"
            out_text = OUTPUT_TEXT_FMT.format(timestep, rewards_text, src_text, trg_text)
            outfile.write(out_text)                       # Write the output to the StringIO stream
            return outfile.getvalue()

    def _check_token(self, tokens, labels):
        return tokens in self._reference_tokens_list

    def _check_token_len(self, tokens):
        return 2 < len(tokens) < self._max_token_num

    def _check_done(self) -> bool:
        return self._success or (not self._token_len_in_range)

    def close(self):
        del self.data       # Data cleanup


class GECEnvGLEU(GECEnvLevDist):

    def __init__(self, *, adapt=True, **kwargs):
        super(GECEnvGLEU, self).__init__(**kwargs)
        self.adapt = adapt

    def _compute_reward(self, prev_tokens, tokens, references) -> float:
        prev_ref_score = adaptive_sentence_gleu(references, prev_tokens, adapt=self.adapt)
        cur_ref_score = adaptive_sentence_gleu(references, tokens, adapt=self.adapt)
        return cur_ref_score - prev_ref_score


# OpenAI Gym environment registrations
register(
        id="gec_lev_dist-v0",
        entry_point="src.envs:GECEnvLevDist",
        max_episode_steps=MAX_EPISODE_STEPS,
        kwargs={
            "label_path": DEFAULT_LABELS_PATH,
        }
)

register(
        id="gec_gleu-v0",
        entry_point="src.envs:GECEnvGLEU",
        max_episode_steps=MAX_EPISODE_STEPS,
        kwargs={
            "label_path": DEFAULT_LABELS_PATH,
        }
)

register(
        id="wi_locness_gec_lev_dist-v0",
        entry_point="src.envs:GECEnvLevDist",
        max_episode_steps=MAX_EPISODE_STEPS,
        kwargs={
            "label_path": DEFAULT_LABELS_PATH,
            "datasets": ["wi+locness"],
        }
)


register(
        id="wi_locness_gec_gleu-v0",
        entry_point="src.envs:GECEnvGLEU",
        max_episode_steps=MAX_EPISODE_STEPS,
        kwargs={
            "label_path": DEFAULT_LABELS_PATH,
            "datasets": ["wi+locness"],
        }
)
