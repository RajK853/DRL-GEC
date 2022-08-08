import os
import numpy as np
from io import StringIO
from gym import Env, spaces, utils
from gym.utils.renderer import Renderer
from gym.envs.registration import register
from nltk.translate.gleu_score import sentence_gleu

from ..utils import decode, load_json, load_text

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DEFAULT_LABELS_PATH = os.path.join(ROOT_PATH, r"data/vocabs/labels.txt")
DEFAULT_REWARD_CONFIG = {
    "gleu_score": 1.0,
    "delay_penalty": -0.1,
    "invalid_label_penalty": -0.2,
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
            data_path,
            max_episode_steps=5,
            render_mode="ansi",
            reward_config=None,
            label_path=DEFAULT_LABELS_PATH,
    ):
        labels = load_text(label_path)
        self.data = load_json(data_path)
        self.num_actions = len(labels)
        self.labels = np.char.array(labels, unicode=True)
        self.action_space = spaces.Discrete(self.num_actions)
        self.observation_space = spaces.Discrete(1)
        self.reward_config = DEFAULT_REWARD_CONFIG.copy()
        if reward_config:
            self.reward_config.update(reward_config)
        self.max_episode_steps = max_episode_steps  # TODO: Maybe not needed?
        # Render configs
        self.render_mode = render_mode
        self.renderer = Renderer(self.render_mode, self._render)
        # Data configs
        self.data_i = 0
        self.num_sents = len(self.data)
        self.data_indexes = np.arange(self.num_sents, dtype="uint32")
        np.random.shuffle(self.data_indexes)

        self.episode_steps = 0
        self.last_tokens = None
        self.last_labels = None
        self.last_rewards = None
        self.current_tokens = None
        self.reference_tokens = None

    def reset(self, *, seed=None, return_info=False, options=None):
        self.episode_steps = 0
        data_index = self.data_indexes[self.data_i]
        data_dict = self.data[data_index]
        self.current_tokens = self.text_to_tokens(data_dict["text"])
        self.reference_tokens = [self.text_to_tokens(ref) for ref in data_dict["references"]]
        self.last_tokens = self.current_tokens
        self.last_labels = ["$KEEP"] * len(self.current_tokens)
        invalid_label_masks = np.zeros(len(self.current_tokens), dtype="uint32")
        self.last_rewards = self.compute_reward(self.last_labels, invalid_label_masks)
        self.data_i += 1
        if self.data_i >= self.num_sents:
            self.data_i = self.data_i % self.num_sents
            np.random.shuffle(self.data_indexes)
        self.renderer.reset()
        self.renderer.render_step()
        return self.current_tokens

    def step(self, actions):
        assert len(self.current_tokens) == len(actions), "Number of current tokens and actions are not same!"
        labels = self.labels[actions]        # Convert label id to label text; 0 -> $KEEP, 1 -> $DELETE, etc
        new_tokens, invalid_label_masks = decode(self.current_tokens, labels)  # Apply the actions
        rewards = self.compute_reward(labels, invalid_label_masks)
        self.last_tokens = self.current_tokens
        self.current_tokens = new_tokens
        self.last_labels = labels
        self.last_rewards = rewards
        self.episode_steps += 1
        done = self._check_done()
        self.renderer.render_step()
        return self.current_tokens, rewards, done, {}

    def _compute_gleu_score(self, labels):
        num_labels = len(labels)
        rewards = np.zeros(num_labels)
        temp_labels = ["$KEEP"] * num_labels  # Temporary labels where all labels are keep except the current label
        for i, label in enumerate(labels):
            temp_labels[i] = label
            out_tokens, _ = decode(self.current_tokens, temp_labels)
            rewards[i] = sentence_gleu(self.reference_tokens, out_tokens)
            temp_labels[i] = "$KEEP"          # Change the label at current index back to $KEEP
        return rewards

    def compute_reward(self, labels, invalid_label_masks):
        gleu_rewards = self.reward_config["gleu_score"]*self._compute_gleu_score(labels)
        delay_penalty = self.reward_config["delay_penalty"]
        invalid_label_penalty = self.reward_config["invalid_label_penalty"]*invalid_label_masks
        return gleu_rewards + delay_penalty + invalid_label_penalty

    def render(self, mode="ansi"):
        if self.render_mode is not None:
            return self.renderer.get_renders()
        else:
            return self._render(mode)

    def _render(self, mode):
        assert mode in self.metadata["render_modes"]
        if mode == "ansi":
            return self._render_text()
        else:
            raise NotImplementedError(f"'{mode}' mode not implemented yet")

    @staticmethod
    def format_tokens(tokens, actions):
        """
        Colorize the updated tokens
        """
        out = []
        for t, a in zip(tokens, actions):
            if a == "$KEEP":
                out.append(t)
            else:
                t = utils.colorize(t, "red")
                a = utils.colorize(a, "blue", bold=True)
                out.append(f"{t} [{a}]")
        return out

    def _render_text(self):
        with StringIO() as outfile:
            rewards = []
            src_tokens = []
            for token, action, reward in zip(self.last_tokens, self.last_labels, self.last_rewards):
                reward = f"{reward:.2f}"
                if action != "$KEEP":
                    token = utils.colorize(token, "green", bold=True)
                    action = utils.colorize(action, "red", bold=True)
                    reward = utils.colorize(reward, "green", bold=True)
                    token = f"{token} [{action}]"
                rewards.append(reward)
                src_tokens.append(token)
            src_text = self.tokens_to_text(src_tokens)
            trg_text = self.tokens_to_text(self.current_tokens)
            rewards_text = ", ".join(rewards)
            out_text = OUTPUT_TEXT_FMT.format(self.episode_steps, rewards_text, src_text, trg_text)
            outfile.write(out_text)
            return outfile.getvalue()

    @staticmethod
    def text_to_tokens(text):
        return text.split()

    @staticmethod
    def tokens_to_text(tokens):
        return " ".join(tokens)

    def _check_done(self):
        step_check = self.episode_steps >= self.max_episode_steps
        return step_check

    def close(self):
        pass


register(
    id="lang8_gec-v0",
    entry_point="src.envs:GECEnv",
    kwargs={
        "data_path": os.path.join(ROOT_PATH, r"data/processed/lang8/data.json"),
    }
)


register(
    id="fce_gec-v0",
    entry_point="src.envs:GECEnv",
    kwargs={
        "data_path": os.path.join(ROOT_PATH, r"data/processed/fce/data.json"),
    }
)


register(
    id="wi_locness_gec-v0",
    entry_point="src.envs:GECEnv",
    kwargs={
        "data_path": os.path.join(ROOT_PATH, r"data/processed/wi+locness/data.json"),
    }
)
