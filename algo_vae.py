import numpy
import torch
import torch
from rl_utils.base import BaseAlgo
from rl_utils.penv import ParallelEnv
from rl_utils import *
from torch_ac.utils import DictList
from torch.distributions.normal import Normal
from model_vae import BeliefVAEModel



class Algo(BaseAlgo):
    """The Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs, env_name, acmodel, algo_name, device=None, num_frames_per_proc=None,
                 discount=0.99, lr=0.001, gae_lambda=0.95, entropy_coef=0.01, value_loss_coef=0.5,
                 max_grad_norm=0.5, recurrence=4, adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256,
                 preprocess_obss=None, reshape_reward=None, history_recurrence=4, batch_size_g=1024,
                 lr_g=0.0003, epochs_g=16, rep_model=None, latent_dim=8, latent_dim_f=16, beta=0.0001, tb_writer=None, seed=-1, use_cnn=False):

        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward)

        self.envs = envs
        self.env_name = env_name
        self.algo_name = algo_name
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.batch_size_g = batch_size_g
        self.epochs_g = epochs_g
        self.rep_model = rep_model
        self.beta = beta
        self.gradient_threshold = 200
        self.latent_dim_f = latent_dim_f

        obs_space, preprocess_obss = rl_utils.get_obss_preprocessor(envs[0].observation_space)

        # Set x_dim and x_size of belief VAE
        if rep_model:
            self.rep_model = rep_model
            predict_full_state = False
            # x_size, x_dim = self.rep_model.latent_dim, 0
            x_size, x_dim = latent_dim_f, latent_dim_f #Hardcode for now
        else:
            predict_full_state = True
            x_size, x_dim = torch.prod(torch.tensor(obs_space["image"])), torch.prod(torch.tensor(obs_space["image"]))  #envs[0].num_objects # 64 pixels with 15 possible values

        # Set belief VAE
        self.belief_vae = BeliefVAEModel(obs_space=obs_space, x_dim=x_dim, x_size=x_size,
                                        latent_dim=latent_dim, predict_full_state=predict_full_state, use_cnn=use_cnn).to(device)

        assert(seed != -1)
        path = "storage/" + str(env_name) +  "_belief_vae_seed" + str(seed)
        status = rl_utils.get_status(path)
        print("Loading model " + path)
        self.belief_vae.load_state_dict(status["vae_model_state"])

        self.history_recurrence = history_recurrence

        assert self.batch_size % self.recurrence == 0

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps)
        self.vae_optimizer = torch.optim.Adam(self.belief_vae.parameters(), lr_g, eps=adam_eps)
        self.batch_num = 0

        #Add the location of the item
        self.item_locations = torch.zeros((num_frames_per_proc, self.num_procs), device=self.device)
        self.has_consulted_genie = torch.zeros((num_frames_per_proc, self.num_procs), device=self.device)
        self.most_recent_consult_step = torch.zeros((num_frames_per_proc, self.num_procs), device=self.device)
        self.step_count = torch.zeros((num_frames_per_proc, self.num_procs), device=self.device)

        # BaseAlgo calls this, but we have a custom ParallelEnv
        self.env = ParallelEnv(envs)
        self.obs = self.env.reset()

        shape = (self.num_frames_per_proc, self.num_procs)

        self.memory = torch.zeros(shape[1], self.belief_vae.memory_size, device=self.device)
        self.memories = torch.zeros(*shape, self.belief_vae.memory_size, device=self.device)

        self.states = [None] * (shape[0])
        self.state = self.env.get_state()

        self.vae_samples = torch.zeros((self.num_frames_per_proc, self.num_procs, 30, latent_dim_f), device=self.device)

        self.values_state_history = torch.zeros(*shape, device=self.device)
        self.tb_writer = tb_writer

        #dummy_data = torch.load("dummy_boxes3_latent16.pt", map_location=device)
        self.test_data, self.test_means = None, None #  dummy_data["data"], dummy_data["means"]
        self.f_latent_dim = x_dim
        self.latent_dim = latent_dim

        self.act_shape = envs[0].action_space.shape

    def get_f_encoding(self, sb):
        states = sb.state.image
        state_features = torch.zeros((states.shape[0], self.f_latent_dim)).to(self.device)
        nonzero_idx = (states[:, :, :, 2] == 1).nonzero()[:, 0]
        idx0 = (states[:, :, :, 2] == 1).nonzero()[:, 1] - 1 #i
        idx1 = (states[:, :, :, 2] == 1).nonzero()[:, 2] - 1 #j
        idx2 = states[torch.arange(idx0.shape[0]), idx0+1, idx1+1, 3] #direction
        idx3 = sb.item_location.to(torch.int64)[nonzero_idx]

        state_features[nonzero_idx] = self.test_data[idx0.to(torch.int64), idx1.to(torch.int64), idx2.to(torch.int64), idx3]  # 512x16 (maybe 512x1)
        return state_features

    def update_g_parameters(self, exps):

        log_grad_norms = []
        log_elbo = []

        with torch.no_grad():
            exps.rep_encodings_mean, _ = self.rep_model.encode_state(exps.state)

        for k in range(self.epochs_g):
            for inds in self._get_batches_starting_indexes(self.history_recurrence, batch_size=self.batch_size_g):
                batch_elbo_loss = 0
                memory = exps.memory[inds]

                for i in range(self.history_recurrence):
                    # Create a sub-batch of experience
                    sb = exps[inds + i]

                    with torch.no_grad():
                        if True:
                            state_features = exps.rep_encodings_mean[inds + i]
                            # state_features = Normal(rep_encoder_mean, rep_encoder_std).sample()
                        else:
                            state_features = self.get_f_encoding(sb)

                    prior_dist = Normal(0, 1)
                    history_encoding, memory = self.belief_vae(sb.obs.image, memory * sb.mask)
                    encoder_mean, encoder_std = self.belief_vae.encoder_dist(state_features, history_encoding)
                    zs = encoder_mean + torch.randn_like(encoder_mean) * encoder_std
                    decoder_mean, decoder_std = self.belief_vae.decoder_dist(zs, history_encoding)

                    elbo = prior_dist.log_prob(zs).sum(dim=-1) + Normal(decoder_mean, decoder_std).log_prob(
                        state_features).sum(dim=-1) - Normal(encoder_mean, encoder_std).log_prob(zs).sum(dim=-1)
                    batch_elbo_loss += -elbo.mean()

                batch_elbo_loss /= self.history_recurrence

                self.vae_optimizer.zero_grad()
                batch_elbo_loss.backward()

                grad_norm = sum(
                    p.grad.data.norm(2).item() ** 2 for p in self.belief_vae.parameters() if p.grad is not None) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.belief_vae.parameters(), self.gradient_threshold)

                self.vae_optimizer.step()

                # Logging
                log_grad_norms.append(grad_norm)
                log_elbo.append(batch_elbo_loss.item())

        logs = {"grad_norm": numpy.mean(log_grad_norms), "batch_elbo_loss": numpy.mean(log_elbo), "starting_elbo": log_elbo[0]}

        return logs


    def update_parameters(self, exps):
        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            for inds in self._get_batches_starting_indexes(self.recurrence):

                # Initialize batch values
                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0
                batch_value_state_history_loss = 0

                # Initialize memory
                if self.acmodel.recurrent:
                    memory = exps.memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience

                    sb = exps[inds + i]

                    dist, value, _, value_state_history = self.acmodel(sb.obs.image, None, sb.vae_samples)

                    entropy = dist.entropy().mean()

                    delta_log_prob = dist.log_prob(sb.action) - sb.log_prob
                    if (len(self.act_shape) == 1):  # Not scalar actions (multivariate)
                        delta_log_prob = torch.sum(delta_log_prob, dim=1)
                    ratio = torch.exp(delta_log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()
                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                    # Update memories for next epoch

                    if self.acmodel.recurrent and i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_value_state_history_loss /= self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic
                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters()) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update log values
                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)


        # Log some values
        logs = {
            "entropy": numpy.mean(log_entropies),
            "value": numpy.mean(log_values),
            "policy_loss": numpy.mean(log_policy_losses),
            "value_loss": numpy.mean(log_value_losses),
            "grad_norm": numpy.mean(log_grad_norms)
        }

        return logs

    def _get_batches_starting_indexes(self, recurrence, batch_size=None):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """
        if batch_size is None:
            batch_size = self.batch_size

        indexes = numpy.arange(0, self.num_frames, recurrence)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + recurrence) % self.num_frames_per_proc != 0]
            indexes += recurrence // 2
        self.batch_num += 1

        num_indexes = batch_size // recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes


    def collect_experiences(self, random_action=False):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """
        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction

            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)

            with torch.no_grad():
                history_encoding, memory = self.belief_vae(preprocessed_obs.image, self.memory * self.mask.unsqueeze(1))
                samples = self.belief_vae.sample(history_encoding)

                dist, value, _, value_state_history = self.acmodel(preprocessed_obs.image, None, samples)

            if random_action is False:
                action = dist.sample()
            else:
                actions_list = [self.env.action_space.sample() for i in range(self.num_procs)]
                action = torch.tensor(actions_list).to(self.device)

            obs, reward, done, _ = self.env.step(action.cpu().numpy())
            state = self.env.get_state()

            # Update experiences values

            self.obss[i] = self.obs
            self.obs = obs

            self.states[i] = self.state
            self.state = state

            self.vae_samples[i] = samples

            if self.acmodel.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            self.values_state_history[i] = value_state_history
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)

            if "Genie" in self.envs[0].__class__.__name__:
                self.item_locations[i] = torch.tensor(self.env.getItemLocation(), device=self.device)
                self.has_consulted_genie[i] = torch.tensor(self.env.hasConsultedGenie(), device=self.device)
                self.step_count[i] = torch.tensor(self.env.getStepCount(), device=self.device)
                self.most_recent_consult_step[i] = torch.tensor(self.env.mostRecentConsultStep(), device=self.device)

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        # preprocessed_state = self.preprocess_obss(self.state, device=self.device)
        with torch.no_grad():
            history_encoding, memory = self.belief_vae(preprocessed_obs.image, self.memory * self.mask.unsqueeze(1))
            samples = self.belief_vae.sample(history_encoding)
            _, next_value, _, next_value_state_history = self.acmodel(preprocessed_obs.image, None, samples)

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0
            next_value_state_history = self.values_state_history[i+1] if i < self.num_frames_per_proc - 1 else next_value_state_history

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]

        exps.state = [self.states[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]

        if self.acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)

        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape((-1, ) + self.action_space_shape)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.value_state_history = self.values_state_history.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape((-1, ) + self.action_space_shape)

        if "Genie" in self.envs[0].__class__.__name__:
            exps.item_location = self.item_locations.transpose(0, 1).reshape(-1)
            exps.has_consulted_genie = self.has_consulted_genie.transpose(0, 1).reshape(-1)
            exps.step_count = self.step_count.transpose(0, 1).reshape(-1)
            exps.most_recent_consult_step = self.most_recent_consult_step.transpose(0, 1).reshape(-1)
        
        exps.vae_samples = self.vae_samples.transpose(0, 1).reshape(-1, 30, self.latent_dim_f)
        
        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)
        exps.state = self.preprocess_obss(exps.state, device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs
