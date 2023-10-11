import torch
import torch
import rl_utils
import argparse
from tqdm import tqdm


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()

    ## General parameters
    parser.add_argument("--env", default='EscapeRoom-8x8-v0 ', required=True,
                        help="name of the environment to train on (REQUIRED)")
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--episodes", type=int, default=1,
                        help="number of episodes (default: 1)")

    args = parser.parse_args()

    num_episodes = args.episodes


    # Collect without parallelization
    env = rl_utils.make_env(args.env, args.seed)
    obs_space, preprocess_obss = rl_utils.get_obss_preprocessor(env.observation_space)
    max_episode_length = env.max_steps

    obss = torch.zeros((num_episodes, max_episode_length, 1, obs_space['image'][0], obs_space['image'][1]))
    states = torch.zeros((num_episodes, max_episode_length, 17))
    rewards = torch.zeros((num_episodes, max_episode_length))
    actions = torch.zeros((num_episodes, max_episode_length, 2))
    masks = torch.zeros((num_episodes, max_episode_length))

    for episode in tqdm(range(num_episodes)):
        obs = env.reset()
        state = env.get_state()
        done = False
        step = 0
        returnn = 0
        while not done and step < max_episode_length:
            # Add to torch tensors
            obss[episode, step] = torch.tensor(obs)
            states[episode, step] = torch.tensor(state)
            action = env.action_space.sample()
            actions[episode, step] = torch.tensor(action)
            masks[episode, step] = 1 - done # The first 0 is where we no longer have data

            obs, reward, done, _ = env.step(action)
            state = env.get_state()

            rewards[episode, step] = reward
            returnn += reward

            step += 1


    # Save pytorch tensor
    torch.save({"obss": obss,
                "states": states,
                "actions": actions,
                "masks": masks,
                "rewards": rewards,
                }, f'collect_{args.env}.pt')


