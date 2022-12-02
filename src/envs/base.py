import os
import numpy as np
from io import StringIO
from gym import Env, spaces, utils
from gym.utils.renderer import Renderer
from typing import List, Dict, Tuple, Any, Union

from src.tokenizers import Tokenizer, WSTokenizer
from src.sampler import IndexSampler, EditMaskGenerator
from src.utils import START_TOKEN, apply_labels, load_json, load_text, filter_correct, filter_by_num_ref


# Global variables
Tokens = List[str]
Actions = np.ndarray
Labels = np.char.array
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DEFAULT_LABELS_PATH = os.path.join(ROOT_PATH, r"data/vocabs/labels.txt")
MAX_EPISODE_STEPS = 5
DEFAULT_REWARD_CONFIG = {
    "scale": 1.0,
    "correct": 0.1,
    "fn_penalty": -0.1,
    "out_of_range_penalty": -1.0,
}
OUTPUT_TEXT_FMT = """\x1b[37;1mTimestep:\x1b[0m {0}  
\x1b[37;1mRewards:\x1b[0m {1}  
\x1b[37;1mSource:\x1b[0m {2}  
\x1b[37;1mOutput:\x1b[0m {3}  
"""


class BaseGECEnv(Env):
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
        self.data = self._init_data(datasets, correct_examples_percent, min_num_refs)
        # Render configs
        self.render_mode = render_mode
        self.renderer = Renderer(self.render_mode, self._render)
        # Data configs
        self.data_i = 0
        self.num_sents = len(self.data)
        data_indexes = np.arange(self.num_sents, dtype="uint32")
        self.mask_generator = EditMaskGenerator(self.labels)
        self.index_sampler = IndexSampler(data_indexes)
        # Environment variables
        self._episode_steps = 0
        self._orig_num_tokens = 0
        self._min_num_tokens = 0
        self._max_num_tokens = 0
        self._are_tokens_correct = False
        self._token_len_out_range = False
        self._tokens_unchanged = True
        self._only_keep_labels = False
        self._success = False
        self._prev_tokens = None
        self._prev_actions = None
        self._prev_labels = None
        self._prev_reward = None
        self._done = None
        self._max_token_num = 0
        self._current_tokens = None
        self._reference_tokens_list = None

    @staticmethod
    def _init_data(datasets, correct_examples_percent=None, min_num_refs=None):
        data = []
        if correct_examples_percent is None:
            correct_examples_percent = [1.0] * len(datasets)
        if min_num_refs is None:
            min_num_refs = [1] * len(datasets)
        for i, name in enumerate(datasets):
            data_path = os.path.join(ROOT_PATH, f"data/processed/{name}/data_filtered.json")
            json_data = load_json(data_path)
            print(f"Original number of data in {name}: {len(json_data)}")
            json_data = filter_correct(json_data, correct_examples_percent[i])
            json_data = filter_by_num_ref(json_data, min_refs=min_num_refs[i])
            print(f"Number of data without correct sentences: {len(json_data)}")
            data.extend(json_data)
        assert len(data) > 1, "No data with current setting."
        return data

    @property
    def reference_tokens_list(self):
        return self._reference_tokens_list

    @property
    def current_reference(self):
        ref_i = np.random.randint(0, len(self._reference_tokens_list))
        reference = self._reference_tokens_list[ref_i]
        return reference

    def init_episode_tokens(self, data_dict: Dict[str, str]):
        """
        Tokenize source-reference pair and add start token if needed
        """
        self._current_tokens = self.tokenizer.text2tokens(data_dict["text"])
        self._reference_tokens_list = [self.tokenizer.text2tokens(ref) for ref in data_dict["references"]]
        if self.add_start and self._current_tokens[0] != START_TOKEN:  # Add start-token to the source-reference pair
            self._current_tokens = [START_TOKEN] + self._current_tokens
            self._reference_tokens_list = [[START_TOKEN] + ref for ref in self._reference_tokens_list]

    def reset(self, *, data_dict=None, seed=None, return_info=False, options=None) -> Tokens:
        self.renderer.reset()
        # Select new source-reference pair
        if data_dict is None:
            self.data_i = self.index_sampler.sample()    # Obtain data index
            data_dict = self.data[self.data_i]           # Obtain data dict with source-reference
        self.init_episode_tokens(data_dict)
        self._max_token_num = max(1.5*len(self._current_tokens), 10)   # Maximum number of allowed tokens
        # Initialize the episode variables
        self._episode_steps = 0
        self._are_tokens_correct = False
        self._token_len_out_range = False
        self._tokens_unchanged = False
        self._only_keep_labels = False
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
        self._are_tokens_correct = self._check_token_in_reference(new_tokens)
        self._token_len_out_range = self._check_token_len(new_tokens)
        self._tokens_unchanged = (self._current_tokens == new_tokens)
        self._only_keep_labels = self._check_all_keep(labels)
        self._success = self._tokens_unchanged and self._are_tokens_correct
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
        raise NotImplementedError

    def compute_reward(self, prev_tokens, tokens, references, labels) -> float:
        reward_scale = self.reward_config["scale"]
        if prev_tokens == tokens:          # Tokens did not change
            if self._check_token_in_reference(tokens) and self._check_all_keep(labels):
                # Tokens are correct and labels are all $KEEP
                reward = self.reward_config["correct"]
            else:
                # Either tokens are not correct or labels are not all $KEEP i.e. some are $UNKNOWN
                reward = self.reward_config["fn_penalty"]
            return reward_scale*reward
        if self._check_token_len(tokens):
            return reward_scale*self.reward_config["out_of_range_penalty"]
        reward = self._compute_reward(prev_tokens, tokens, references=references)
        return reward_scale*reward

    def render(self, mode: str = "ansi") -> Union[list, None, str]:
        if self.render_mode is not None:
            return self.renderer.get_renders()
        else:
            return self._render(mode)

    def _render(self, mode: str = "ansi") -> str:
        assert mode in self.metadata["render_modes"]
        if mode == "ansi":
            return self.render_text(
                    prev_tokens=self._prev_tokens,
                    labels=self._prev_labels,
                    reward=self._prev_reward,
                    current_tokens=self._current_tokens,
                    episode_step=self._episode_steps,
            )
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

    def _check_token_in_reference(self, tokens):
        return tokens in self._reference_tokens_list

    def _check_token_len(self, tokens):
        num_tokens = len(tokens)
        return (num_tokens <= 2) or (num_tokens >= self._max_token_num)

    @staticmethod
    def _check_all_keep(labels):
        return all(labels == "$KEEP")

    def _check_done(self) -> bool:
        return self._tokens_unchanged or self._token_len_out_range

    def close(self):
        del self.data       # Data cleanup
