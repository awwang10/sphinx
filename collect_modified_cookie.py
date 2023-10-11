import torch
import torch.nn.functional as F
import torch
import rl_utils
import argparse
import pdb
from tqdm import tqdm
from gym_minigrid.cookie_modified import BeliefTracker


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()

    ## General parameters
    parser.add_argument("--env", required=True,
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

    obss = torch.zeros((num_episodes, max_episode_length, obs_space['image'][0], obs_space['image'][1], obs_space['image'][2]))
    states = torch.zeros((num_episodes, max_episode_length, obs_space['image'][0], obs_space['image'][1], obs_space['image'][2]))
    rewards = torch.zeros((num_episodes, max_episode_length))
    actions = torch.zeros((num_episodes, max_episode_length))
    masks = torch.zeros((num_episodes, max_episode_length))
    cookie_locations = torch.zeros((num_episodes, max_episode_length, obs_space['image'][0], obs_space['image'][1]))
    button_locations = torch.zeros((num_episodes, max_episode_length, obs_space['image'][0], obs_space['image'][1]))

    room_belief = torch.zeros((num_episodes, max_episode_length))
    cookie_belief = torch.zeros((num_episodes, max_episode_length))

    for episode in tqdm(range(num_episodes)):
        obs = env.reset()
        state = env.get_state()

        done = False
        step = 0
        returnn = 0

        belief_tracker = BeliefTracker(env)

        while not done and step < max_episode_length:
            # Add to torch tensors
            obss[episode, step] = torch.tensor(obs["image"])
            states[episode, step] = torch.tensor(state["image"])
            action = env.action_space.sample()
            actions[episode, step] = action
            masks[episode, step] = 1 - done # The first 0 is where we no longer have data
            for pos in env.cookie_positions:
                cookie_locations[episode, step, pos[0], pos[1]] = 1

            button_locations[episode, step, env.button_position[0], env.button_position[1]] = 1

            obs, reward, done, _ = env.step(action)
            state = env.get_state()

            belief_tracker.update(env)
            room_belief[episode, step] = int(belief_tracker.status, 2)
            cookie_belief[episode, step] = int(belief_tracker.cookie, 2)

            rewards[episode, step] = reward
            returnn += reward

            step += 1

    # Save pytorch tensor
    torch.save({"obss": obss,
                "states": states,
                "actions": actions,
                "masks": masks,
                "rewards": rewards,
                "cookie_locations": cookie_locations,
                "button_locations": button_locations,
                "room_belief": room_belief,
                "cookie_belief": cookie_belief,
                }, f'collect_{args.env}.pt')


