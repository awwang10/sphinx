import torch

import rl_utils
from .other import device
from model_policy import ACModel

class Agent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, obs_space, action_space, model_dir, algo_name="belief_vae", argmax=False, num_envs=1):

        obs_space, self.preprocess_obss = rl_utils.get_obss_preprocessor(obs_space)

        acmodel = ACModel(obs_space, action_space, aux_info_embedding_size=0, algo_name=algo_name, use_memory=None, use_text=None, aux_info=False)
        acmodel.load_state_dict(rl_utils.get_model_state(model_dir) )
        acmodel.to(device)
        self.acmodel = acmodel
        self.argmax = argmax
        self.num_envs = num_envs


    def perform_action(self, obss):
        preprocessed_obss = self.preprocess_obss(obss, device=device)

    def get_actions(self, obs, history_encoding, samples):
        preprocessed_obs = self.preprocess_obss(obs, device=device)
        dist, value, _, value_state_history = self.acmodel(preprocessed_obs, None, samples)

        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        return actions.cpu().numpy()

    def get_action(self, obss, history_encoding, samples):
        return self.get_actions([obss], [history_encoding], samples)[0]

    def analyze_feedbacks(self, rewards, dones):
        pass

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])
