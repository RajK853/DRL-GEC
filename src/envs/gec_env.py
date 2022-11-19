import os
from typing import List
from gym.envs.registration import register

from .base import BaseGECEnv, Tokens
from src.utils import get_lev_dist, lev_dist_reward, adaptive_sentence_gleu


# Global variables
MAX_EPISODE_STEPS = 5
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DEFAULT_LABELS_PATH = os.path.join(ROOT_PATH, r"data/vocabs/labels.txt")
LD_V0_REWARD_CONFIG = {
    "correct": 0.1,
    "fn_penalty": 0.0,
    "out_of_range_reward": -1.0,
}
LD_V1_REWARD_CONFIG = {
    "correct": 1.0,
    "fn_penalty": -2.0,
    "out_of_range_reward": -2.0,
}


class GECEnvLevDist(BaseGECEnv):

    @staticmethod
    def _compute_reward(prev_tokens: Tokens, tokens: Tokens, references: List[Tokens]) -> float:
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


class GECEnvLevDistV1(BaseGECEnv):

    @staticmethod
    def _compute_reward(prev_tokens: Tokens, tokens: Tokens, references: List[Tokens]) -> float:
        rewards = []
        for ref_tokens in references:
            len_ref = len(ref_tokens)
            prev_ref_lev_dist = get_lev_dist(prev_tokens, ref_tokens)
            cur_ref_lev_dist = get_lev_dist(tokens, ref_tokens)
            lev_reward = prev_ref_lev_dist - cur_ref_lev_dist
            rewards.append(lev_reward)
        return max(rewards)


class GECEnvGLEU(BaseGECEnv):

    def __init__(self, *, adapt: bool = True, **kwargs):
        super(GECEnvGLEU, self).__init__(**kwargs)
        self.adapt = adapt

    def _compute_reward(self, prev_tokens: Tokens, tokens: Tokens, references: List[Tokens]) -> float:
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
            "reward_config": LD_V0_REWARD_CONFIG,
        }
)


register(
        id="gec_lev_dist-v1",
        entry_point="src.envs:GECEnvLevDistV1",
        max_episode_steps=MAX_EPISODE_STEPS,
        kwargs={
            "label_path": DEFAULT_LABELS_PATH,
            "reward_config": LD_V1_REWARD_CONFIG,
        }
)

register(
        id="wi_locness_gec_lev_dist-v0",
        entry_point="src.envs:GECEnvLevDist",
        max_episode_steps=MAX_EPISODE_STEPS,
        kwargs={
            "label_path": DEFAULT_LABELS_PATH,
            "datasets": ["wi+locness"],
            "reward_config": LD_V0_REWARD_CONFIG,
        }
)

register(
        id="wi_locness_gec_lev_dist-v1",
        entry_point="src.envs:GECEnvLevDistV1",
        max_episode_steps=MAX_EPISODE_STEPS,
        kwargs={
            "label_path": DEFAULT_LABELS_PATH,
            "datasets": ["wi+locness"],
            "reward_config": LD_V0_REWARD_CONFIG,
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
        id="wi_locness_gec_gleu-v0",
        entry_point="src.envs:GECEnvGLEU",
        max_episode_steps=MAX_EPISODE_STEPS,
        kwargs={
            "label_path": DEFAULT_LABELS_PATH,
            "datasets": ["wi+locness"],
        }
)
