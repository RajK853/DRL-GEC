import os
import sys
import gym
import torch
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime
from collections import defaultdict
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from nltk.translate.gleu_score import corpus_gleu

import src.envs
from src.algos import reinforce
from src.search import search_best_actions
from src.utils import remove_ansi, iterative_prediction, load_yaml, load_json, write_json, discount_cumsum, load_model

gym.logger.set_level(40)             # Disable displaying warnings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def select_action(policy, state, reference, mask_generator):
    [logits] = policy([state])
    dist = Categorical(logits=logits)
    action_np = search_best_actions(policy, state, reference, mask_generator)
    action = torch.from_numpy(action_np).to(logits.device)
    log_pi = dist.log_prob(action)
    return action_np, log_pi, dist.entropy()


def get_evaluator(datasets, label_vocab, num_iterations=10):
    sources = ()
    references = ()
    for dataset in datasets:
        data_path = f"data/processed/{dataset}/dev.json"
        json_data = load_json(data_path)
        src_ref = ((data_dict["text"], data_dict["references"]) for data_dict in json_data)
        src_data, ref_data = zip(*src_ref)
        sources += src_data
        references += ref_data
        del json_data

    print(f"Number of evaluation examples: {len(sources)}")

    def eval_func(policy):
        with torch.cuda.amp.autocast():
            policy.eval()          # Enable evaluation model i.e. disables dropout
            predictions = iterative_prediction(
                    policy, label_vocab, sources, num_iter=num_iterations, insert_start=True, verbose=False
            )
            score = corpus_gleu(references, predictions)
            policy.train()
            return score

    return eval_func


def main(
        lr,
        gamma,
        batch_size,
        update_interval,
        dropout,
        episodes,
        eval_max_iter,
        evaluate_interval,
        record_output_interval,
        model_path,
        log_dir,
        env_kwargs,
        meta_data,
):
    train_type = "pretrain" if model_path is None else "finetune"
    current_datetime = datetime.now().strftime("%d_%m_%Y_%H:%M")
    exp_log_dir = os.path.join(log_dir, f"{train_type}_rl_{current_datetime}")

    env = gym.make(new_step_api=True, **env_kwargs)
    model_name = "roberta-base"
    tokenizer_config = {"use_fast": True}
    transformer_config = {"output_attentions": False}
    policy = load_model(
            model_name=model_name,
            model_path=model_path,
            num_labels=env.action_space.n,
            tokenizer_config=tokenizer_config,
            transformer_config=transformer_config,
            local_files_only=True,
    ).to(device)
    policy.train()
    optim = torch.optim.Adam(policy.parameters(), lr=lr)
    grad_scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter(log_dir=exp_log_dir)
    evaluator = get_evaluator(env_kwargs["datasets"], env.labels, eval_max_iter)
    write_json(os.path.join(exp_log_dir, "meta.json"), meta_data)

    # Log hyperparameters
    writer.add_scalar("hyperparameters/gamma", gamma, 0)
    writer.add_scalar("hyperparameters/dropout", dropout, 0)
    writer.add_scalar("hyperparameters/batch_size", batch_size, 0)
    writer.add_scalar("hyperparameters/update_interval", update_interval, 0)
    # Variables for training progress bars
    policy_pbar = None
    max_eval_score = 0
    eval_score = evaluator(policy)
    writer.add_scalar("rl/eval_score", eval_score, 1)
    buffer_dict = defaultdict(list)
    for episode in tqdm(range(1, episodes + 1), desc="Training Episodes", total=episodes):
        rewards = []
        log_pis = []
        entropies = []
        token_lens = []
        done = False
        init_state = state = env.reset()
        reference = env.current_reference
        # explore = np.random.uniform() < eps
        with torch.cuda.amp.autocast():
            while not done:
                action, log_pi, entropy = select_action(policy, state, reference, env.mask_generator)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                # Save timestep data
                rewards.append(reward)
                log_pis.append(log_pi)
                entropies.append(entropy)
                token_lens.append(len(next_state))
                buffer_dict["state"].append(state)
                buffer_dict["action"].append(action)
                state = next_state
        # Compute returns
        rewards = np.array(rewards)
        returns = discount_cumsum(rewards, discount=gamma)
        buffer_dict["return"].extend(returns)
        # eps = max(eps*eps_decay, min_eps)
        # Train the model
        if (episode % update_interval) == 0:
            if policy_pbar is None:
                policy_pbar = tqdm(desc="Updating Policy")
            loss = reinforce.train(policy_pbar, optim, grad_scaler, policy, buffer_dict, batch_size=batch_size)
            writer.add_scalar("rl/mean_loss", loss, episode)
            buffer_dict = defaultdict(list)
            torch.cuda.empty_cache()
        # Log the episode output to the tensorboard
        if (episode % record_output_interval) == 0:
            render_output = "  \n".join(remove_ansi(out) for out in env.render())
            writer.add_text("rl/output", render_output, episode)
        # Evaluate the model
        if (episode % evaluate_interval) == 0:
            eval_score = evaluator(policy)
            if eval_score >= max_eval_score:
                torch.save(policy.state_dict(), os.path.join(exp_log_dir, "model-best.pt"))
                max_eval_score = eval_score
            writer.add_scalar("rl/eval_score", eval_score, episode)
        # Log scalar episode results
        writer.add_scalar("rl/lr", lr, episode)
        # writer.add_scalar("rl/eps", eps, episode)
        # writer.add_scalar("rl/explore", explore, episode)
        writer.add_scalar("rl/episode_length", len(rewards), episode)
        writer.add_scalar("rl/episode_reward_last", rewards[-1], episode)
        writer.add_scalar("rl/episode_reward_total", rewards.sum(), episode)
        writer.add_scalar("rl/token_length_delta_ratio", (len(state) - len(init_state)) / len(init_state), episode)
        # Log histogram episode results
        writer.add_histogram("rl/episode_reward", rewards, episode)
        writer.add_histogram("rl/episode_returns", returns, episode)
        writer.add_histogram("rl/episode_log_pi", torch.cat(log_pis), episode)
        writer.add_histogram("rl/episode_entropy", torch.cat(entropies), episode)
        writer.add_histogram("rl/episode_token_length", np.array(token_lens), episode)
    torch.save(policy.state_dict(), os.path.join(exp_log_dir, "model-last.pt"))


if __name__ == "__main__":
    config_path = sys.argv[1]
    params = load_yaml(config_path)
    main(**params)
