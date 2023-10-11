import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from rl_utils.other import device

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class RepresentationModel(nn.Module):

    def __init__(self, obs_space, action_space, use_cnn, latent_dim=16):
        super().__init__()

        self.recurrent = 1          # Needed by BaseAlgo class
        self.aux_info = None        # Needed by BaseAlgo class

        self.latent_dim = latent_dim
        print(latent_dim, "Representation Model Latent Dim")

        self.use_cnn = use_cnn

        if use_cnn:
            self.image_embedding_size = 192

            self.image_cnn = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=8, kernel_size=9, stride=4),
                nn.ReLU(),
                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(512, self.image_embedding_size)
            )
            
        else:
            n = obs_space["image"][0]
            self.width = n
            m = obs_space["image"][1]
            self.height = m
            k = obs_space["image"][2]
            self.image_embedding_size = n * m * k

         # Define auxiliary critic V(s,o) model

        self.critic = nn.Sequential(
            nn.Linear(self.image_embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        if self.use_cnn:
            action_space_dim = 4
        else:
            action_space_dim = action_space.n

        self.actor = nn.Sequential(
            nn.Linear(self.image_embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_space_dim)
        )

        if self.use_cnn:
            dynamics_input_shape = 2 * self.latent_dim + action_space.shape[0]
        else:
            dynamics_input_shape = 2*self.latent_dim + 1

        self.dynamics_model = nn.Sequential(
            nn.Linear(dynamics_input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )


        self.next_state_model = nn.Linear(128, self.latent_dim)
        self.next_obs_model = nn.Linear(128, self.latent_dim)
        self.reward_model = nn.Linear(128, 1)

        if self.use_cnn:
            state_input_size = 17
        else:
            state_input_size = self.image_embedding_size

        self.state_encoder = nn.Sequential(
            nn.Linear(state_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2*latent_dim)
        )

        self.obs_encoder = nn.Sequential(
            nn.Linear(self.image_embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2*latent_dim)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def image_decoder(self, s):
        return s.reshape(s.shape[0], -1)

    def encode_state(self, state):
        if not self.use_cnn: # all env's except escape room
            s = state.image.transpose(1, 3).transpose(2, 3)
            s = self.image_decoder(s.contiguous())
        else:
            s = state.image
        output = self.state_encoder(s)
        encoder_mean = output[:, :self.latent_dim]
        encoder_std = F.softplus(output[:, self.latent_dim:], threshold=1) + 1e-5
        return encoder_mean, encoder_std

    def encode_obs(self, obs):
        if not self.use_cnn:  # all env's except escape room
            o = obs.image.transpose(1, 3).transpose(2, 3)
            o = self.image_decoder(o.contiguous())
        else:
            o = self.image_cnn(obs.image)
        output = self.obs_encoder(o)
        encoder_mean = output[:, :self.latent_dim]
        encoder_std = F.softplus(output[:, self.latent_dim:], threshold=1) + 1e-5
        return encoder_mean, encoder_std

    def predict_next(self, state, obs, action):
        encoder_mean_s, encoder_std_s = self.encode_state(state)
        zs = encoder_mean_s + torch.randn_like(encoder_mean_s) * encoder_std_s

        encoder_mean_o, encoder_std_o = self.encode_obs(obs)
        zo = encoder_mean_o + torch.randn_like(encoder_mean_o) * encoder_std_o

        if len(action.shape) == 1:
            action = action.float().unsqueeze(dim=1)
        else:
            action = action.float()
        embedding = self.dynamics_model(torch.cat([zs, zo, action], dim=1))
        return self.next_state_model(embedding), self.next_obs_model(embedding), self.reward_model(embedding)

    def forward(self, state, obs):
        s = state.image.transpose(1, 3).transpose(2, 3)
        s = self.image_decoder(s.contiguous())

        actions = self.actor(s)
        dist = Categorical(logits=F.log_softmax(actions, dim=1))

        value = self.critic(s).squeeze(1)
        return value, dist
