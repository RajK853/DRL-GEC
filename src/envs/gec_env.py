import os
import numpy as np
from io import StringIO
from gym import Env, spaces, utils
from gym.utils.renderer import Renderer
from typing import List, Dict, Tuple, Any, Union
from gym.envs.registration import register
from nltk.translate.gleu_score import sentence_gleu

from ..sampler import IndexSampler
from ..tokenizers import Tokenizer, WSTokenizer
from ..utils import decode, load_json, load_text, START_TOKEN

# Global variables
Tokens = List[str]
Actions = np.ndarray
Labels = np.char.array
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DEFAULT_LABELS_PATH = os.path.join(ROOT_PATH, r"data/vocabs/labels.txt")
MAX_EPISODE_STEPS = 5
DEFAULT_REWARD_CONFIG = {
    "gleu_score": 1.0,
    "delay_penalty": -0.1,
    "invalid_label_penalty": -0.5,
}
OUTPUT_TEXT_FMT = """\x1b[37;1mTimestep:\x1b[0m {0}
\x1b[37;1mRewards:\x1b[0m {1}
\x1b[37;1mSource:\x1b[0m {2}
\x1b[37;1mOutput:\x1b[0m {3}
"""


class GECEnv(Env):
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
    }

    def __init__(
            self,
            data_path: str,
            label_path: str = DEFAULT_LABELS_PATH,
            render_mode: str = "ansi",
            max_episode_steps: int = MAX_EPISODE_STEPS,
            tokenizer: Tokenizer = WSTokenizer,
            reward_config: Dict[str, float] = None,
            add_start: bool = True,
    ):
        self.tokenizer = tokenizer
        self.add_start = add_start
        self.max_episode_steps = max_episode_steps
        self.reward_config = DEFAULT_REWARD_CONFIG.copy()
        if reward_config:
            self.reward_config.update(reward_config)
        labels = load_text(label_path)
        self.data = load_json(data_path)
        self.num_actions = len(labels)
        self.labels = np.char.array(labels, unicode=True)
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Discrete(1)
        # Render configs
        self.render_mode = render_mode
        self.renderer = Renderer(self.render_mode, self._render)
        # Data configs
        self.data_i = 0
        self.num_sents = len(self.data)
        data_indexes = np.arange(self.num_sents, dtype="uint32")
        self.index_sampler = IndexSampler(data_indexes, interval=32, repeat=4)
        # Environment variables
        self._episode_steps = 0
        self._orig_num_tokens = 0
        self._min_num_tokens = 0
        self._max_num_tokens = 0
        self._prev_tokens = None
        self._prev_labels = None
        self._prev_reward = None
        self._current_tokens = None
        self._reference_tokens_list = None

    def reset(self, *, seed=None, return_info=False, options=None) -> Tokens:
        # Select new source-reference pair
        self.data_i = self.index_sampler.sample()    # Obtain data index
        data_dict = self.data[self.data_i]           # Obtain data dict with source-reference
        # Tokenize source-reference pair
        self._current_tokens = self.tokenizer.text2tokens(data_dict["text"])
        self._reference_tokens_list = [self.tokenizer.text2tokens(ref) for ref in data_dict["references"]]
        if self.add_start:                           # Add start-token to the source-reference pair
            self._current_tokens = [START_TOKEN] + self._current_tokens
            self._reference_tokens_list = [[START_TOKEN] + ref for ref in self._reference_tokens_list]
        # Initialize the episode variables
        self._episode_steps = 0
        self._orig_num_tokens = len(self._current_tokens)
        self._min_num_tokens = max(2, np.math.ceil(0.5 * self._orig_num_tokens))      # TODO: Remove hard-coding
        self._max_num_tokens = round(1.5 * self._orig_num_tokens)                     # TODO: Remove hard-coding
        self._prev_tokens = self._current_tokens                   # No previous tokens. So set it to current tokens
        self._prev_labels = ["$KEEP"] * self._orig_num_tokens      # No previous labels. So set it to all keeps
        # Compute initial reward
        invalid_label_masks = np.zeros(self._orig_num_tokens, dtype="uint32")
        self._prev_reward = self.compute_reward(self._prev_labels, invalid_label_masks)
        # Initialize the renderer
        self.renderer.reset()
        self.renderer.render_step()
        return self._current_tokens

    def step(self, actions: Actions) -> Tuple[Tokens, float, bool, Dict[str, Any]]:
        assert len(self._current_tokens) == len(actions), "Number of current tokens and actions are not same!"
        labels = self.labels[actions]         # Obtain labels from actions; ["$KEEP", .., "$DELETE"] from [0, .., 1]
        new_tokens, invalid_label_masks = decode(self._current_tokens, labels)  # Apply the labels
        reward = self.compute_reward(labels, invalid_label_masks)
        # Update the environment variables
        self._prev_tokens = self._current_tokens
        self._current_tokens = new_tokens
        self._prev_labels = labels
        self._prev_reward = reward
        self._episode_steps += 1
        done = self._check_done()
        self.renderer.render_step()
        return self._current_tokens, reward, done, {}

    def _compute_gleu_score(self, labels: Labels) -> float:
        new_tokens, invalid_label_masks = decode(self._current_tokens, labels)
        state_gleu = sentence_gleu([self._current_tokens], new_tokens)
        ref_gleu = sentence_gleu(self._reference_tokens_list, new_tokens)
        reward = (state_gleu + 2 * ref_gleu) / 3                          # TODO: Remove hard-coding
        return reward

    def compute_reward(self, labels: Labels, invalid_label_masks: np.ndarray) -> float:
        gleu_rewards = self.reward_config["gleu_score"] * self._compute_gleu_score(labels)
        delay_penalty = self.reward_config["delay_penalty"]
        invalid_label_penalty = self.reward_config["invalid_label_penalty"] * np.mean(invalid_label_masks)
        total_reward = gleu_rewards + delay_penalty + invalid_label_penalty
        return float(total_reward)

    def render(self, mode: str = "ansi") -> Union[list, None, str]:
        if self.render_mode is not None:
            return self.renderer.get_renders()
        else:
            return self._render(mode)

    def _render(self, mode: str = "ansi") -> str:
        assert mode in self.metadata["render_modes"]
        if mode == "ansi":
            return self._render_text()
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

    def _render_text(self) -> str:
        with StringIO() as outfile:
            # Colorize the source tokens based on their labels
            src_tokens = self.format_tokens(self._prev_tokens, self._prev_labels)
            # Colorize the timestep to indicate the end of the episode
            if self._check_done():
                timestep = utils.colorize(str(self._episode_steps), "green", bold=True)
            else:
                timestep = self._episode_steps
            # Convert tokens into text
            src_text = self.tokenizer.tokens2text(src_tokens)
            trg_text = self.tokenizer.tokens2text(self._current_tokens)
            # Obtain output to render
            rewards_text = f"{self._prev_reward:.3f}"
            out_text = OUTPUT_TEXT_FMT.format(timestep, rewards_text, src_text, trg_text)
            outfile.write(out_text)                       # Write the output to the StringIO stream
            return outfile.getvalue()

    def _check_done(self) -> bool:
        # Conditions are created as local functions to organize them and only compute required conditions
        def all_keep():
            return all(label == "$KEEP" for label in self._prev_labels)

        def matches_any_ref():
            return any(self._current_tokens == ref_tokens for ref_tokens in self._reference_tokens_list)

        def invalid_token_len():
            return not (self._min_num_tokens <= len(self._current_tokens) <= self._max_num_tokens)

        def max_steps_reached():
            return self._episode_steps >= self.max_episode_steps

        return max_steps_reached() or invalid_token_len() or (all_keep() and matches_any_ref())

    def close(self):
        del self.data       # Data cleanup


register(
        id="lang8_gec-v0",
        entry_point="src.envs:GECEnv",
        max_episode_steps=MAX_EPISODE_STEPS,
        kwargs={
            "data_path": os.path.join(ROOT_PATH, r"data/processed/lang8/data.json"),
        }
)


register(
        id="fce_gec-v0",
        entry_point="src.envs:GECEnv",
        max_episode_steps=MAX_EPISODE_STEPS,
        kwargs={
            "data_path": os.path.join(ROOT_PATH, r"data/processed/fce/data.json"),
        }
)

register(
        id="wi_locness_gec-v0",
        entry_point="src.envs:GECEnv",
        max_episode_steps=MAX_EPISODE_STEPS,
        kwargs={
            "data_path": os.path.join(ROOT_PATH, r"data/processed/wi+locness/data.json"),
        }
)
