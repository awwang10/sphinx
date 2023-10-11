import torch
from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList
from torch.distributions.normal import Normal
from model_vae import BeliefVAEModel
from rl_utils import *

class Algo():
    """The Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, env, device=None, adam_eps=1e-8, preprocess_obss=None, lr_g=0.0003, epochs_g=16, rep_model=None,
                 latent_dim=8, latent_dim_f=16, beta=0.0001, gradient_threshold=200, tb_writer=None, use_cnn=False):

        self.env = env
        self.device = device
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.epochs_g = epochs_g
        self.latent_dim = latent_dim
        self.gradient_threshold = gradient_threshold

        obs_space, preprocess_obss = rl_utils.get_obss_preprocessor(env.observation_space)

        self.rep_model = rep_model
        predict_full_state = self.rep_model is None

        if predict_full_state:
            self.obs_space = obs_space
            image_pixel_count = obs_space["image"][0] * obs_space["image"][1] * obs_space["image"][2]
            x_size, x_dim = image_pixel_count,image_pixel_count
        else:
            x_size, x_dim = latent_dim_f, latent_dim_f



        # Set belief VAE
        self.belief_vae = BeliefVAEModel(obs_space=obs_space, x_dim=x_dim, x_size=x_size,
                                         latent_dim=latent_dim, predict_full_state=predict_full_state, use_cnn=use_cnn).to(device)

        self.vae_optimizer = torch.optim.Adam(self.belief_vae.parameters(), lr_g, eps=adam_eps)
        self.batch_num = 0

        self.tb_writer = tb_writer

        self.f_latent_dim = 16
        self.test_data, self.test_means = None, None

        self.log_episode = 1

        self.log_episodes = [] #For cookie only



    def get_f_encoding(self, sb):
        if "Genie" in self.env.__class__.__name__:
            states = sb.state
            state_features = torch.zeros((states.shape[0], self.f_latent_dim)).to(self.device)
            nonzero_idx = (states[:, :, :, 2] == 1).nonzero()[:, 0]
            idx0 = (states[:, :, :, 2] == 1).nonzero()[:, 1] - 1 #i
            idx1 = (states[:, :, :, 2] == 1).nonzero()[:, 2] - 1 #j
            idx2 = states[torch.arange(idx0.shape[0]), idx0+1, idx1+1, 3] #direction
            idx3 = sb.item_location.to(torch.int64)[nonzero_idx]

            state_features[nonzero_idx] = self.test_data[idx0.to(torch.int64), idx1.to(torch.int64), idx2.to(torch.int64), idx3]  # 512x16 (maybe 512x1)
            return state_features

        if "Cookie" in self.env.__class__.__name__:
            states = sb.state
            state_features = torch.zeros((states.shape[0], self.f_latent_dim)).to(self.device)
            nonzero_idx = (states[:, 1, 5, 0] == 1 or states[:, 11, 5, 0] == 1).nonzero()[:, 0]


    def get_f(self, item_location):
        return self.test_data[0, 0, 0, item_location]


    def nearest_neighbour(self, latents):
        dists = torch.stack([torch.linalg.norm(latents - self.get_f(0), ord=2, dim=1), torch.linalg.norm(latents - self.get_f(1), ord=2, dim=1), torch.linalg.norm(latents - self.get_f(2), ord=2, dim=1)])
        item_guesses = torch.argmin(dists, dim=0)

        return [(item_guesses==0).sum() / latents.shape[0], (item_guesses==1).sum() / latents.shape[0], (item_guesses==2).sum() / latents.shape[0]]

    def nearest_neighbour_encoded(self, sample_f_encodings, *f_means):

        sample_size = sample_f_encodings.shape[0]
        norms = [torch.linalg.norm(sample_f_encodings - f_mean, ord=2, dim=1) for f_mean in f_means]
        dists = torch.stack(norms)
        item_guesses = torch.argmin(dists, dim=0)

        return [ (item_guesses == i).sum() / sample_size for i in range(sample_size) ]


    def update_g_parameters(self, exps):

        history_encodings = []
        batch_elbo_loss = 0
        max_steps, num_episodes = exps.mask.shape[0], exps.mask.shape[1]

        # Initialize memory
        memory = torch.zeros((num_episodes, self.belief_vae.history_model.memory_size)).to(self.device)

        if "Genie" in self.env.__class__.__name__:
            talk_to_genie = torch.zeros((max_steps, num_episodes))

        if "Cookie" in self.env.__class__.__name__:
            pass

        for step in range(max_steps):
            # Do only if not all episodes have ended
            sb = exps[step]

            if sb.mask.sum() > 0:
                with torch.no_grad():

                    if self.rep_model is not None:
                        rep_encoder_mean, rep_encoder_std = self.rep_model.encode_state(DictList({'image': sb.state.to(device)}))
                        state_features = rep_encoder_mean
                    else:
                        state_features = sb.state.to(device).flatten(start_dim=1)

                prior_dist = Normal(0, 1)

                if step % 16 == 0:
                    history_encoding, memory = self.belief_vae(sb.obs.to(device), memory.detach() * sb.mask.to(device).unsqueeze(dim=1))
                else:
                    history_encoding, memory = self.belief_vae(sb.obs.to(device), memory * sb.mask.to(device).unsqueeze(dim=1))

                encoder_mean, encoder_std = self.belief_vae.encoder_dist(state_features, history_encoding)
                zs = encoder_mean + torch.randn_like(encoder_mean) * encoder_std
                decoder_mean, decoder_std = self.belief_vae.decoder_dist(zs, history_encoding)
                elbo = prior_dist.log_prob(zs).sum(dim=-1) + Normal(decoder_mean, decoder_std).log_prob(state_features).sum(dim=-1) - Normal(encoder_mean, encoder_std).log_prob(zs).sum(dim=-1)
                batch_elbo_loss += -(elbo * sb.mask.to(device)).sum() #* torch.pow(torch.tensor(0.95).to(device), step) #

                # Logging
                if "Genie" in self.env.__class__.__name__:
                    talk_to_genie[step] = (sb.obs.to(device)[:, self.env.genie_location[0], self.env.genie_location[1], 1] < 3) * sb.mask.to(device)

                    for s in range(0, 90, 10): #We log every step of epiosde "self.log_episode"
                        if step == s:
                            history_encodings.append((history_encoding[self.log_episode], step))

                if "Cookie" in self.env.__class__.__name__:
                    for s in range(0, 200, 10):  # We log every step of epiosde "self.log_episode"
                        if step == s:
                            history_encodings.append((history_encoding[self.log_episode], step))

            else:
                break

        batch_elbo_loss /= exps.mask.sum()

        self.vae_optimizer.zero_grad()
        batch_elbo_loss.backward()

        grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.belief_vae.parameters() if p.grad is not None) ** 0.5
#        torch.nn.utils.clip_grad_norm_(self.belief_vae.parameters(), self.gradient_threshold)
        print("Grad norm:", grad_norm)

        self.vae_optimizer.step()

        logs = {"batch_elbo_loss": batch_elbo_loss}
        return logs
