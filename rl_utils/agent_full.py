import torch

import rl_utils
from .other import device
from model_f import RepresentationModel

class FullyObservableAgent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, obs_space, action_space, model_dir, argmax=False, num_envs=1):

        obs_space, self.preprocess_obss = rl_utils.get_obss_preprocessor(obs_space)

        self.rep_model = RepresentationModel(obs_space=obs_space, action_space=action_space).to(device)
        self.rep_model.load_state_dict(rl_utils.get_model_state(model_dir))
        self.rep_model.to(device)
        self.rep_model.eval()
        self.argmax = argmax
        self.num_envs = num_envs

        # if hasattr(self.preprocess_obss, "vocab"):
        #     self.preprocess_obss.vocab.load_vocab(rl_utils.get_vocab(model_dir))


    def perform_action(self, obss):
        preprocessed_obss = self.preprocess_obss(obss, device=device)

    def get_actions(self, states, obss):
        preprocessed_obs = self.preprocess_obss(obss, device=device)
        preprocessed_state = self.preprocess_obss(states, device=device)
        value, dist, encoder_mean, encoder_std = self.rep_model(preprocessed_state, preprocessed_obs)
        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        return actions.cpu().numpy()

    def get_action(self, state, obs):
        return self.get_actions([state], [obs])[0]

    def analyze_feedbacks(self, rewards, dones):
        pass

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])
