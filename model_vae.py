import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from rl_utils.other import device


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class ImageEncoder(nn.Module):
    def __init__(self, obs_space):
            super().__init__()

            input_size = obs_space['image'][0] * obs_space['image'][1] * obs_space['image'][2]

            self.embedding_size = 512 # image_embedding size

            self.image_conv = nn.Sequential(
                nn.Linear(input_size, self.embedding_size),
            )
            # n = obs_space["image"][0]
            # m = obs_space["image"][1]
            # self.linear = nn.Linear(((n - 1) // 2 - 2) * ((m - 1) // 2 - 2) * 64, self.embedding_size)

    def forward(self, obs):
        if torch.is_tensor(obs):
            # x = obs.reshape(obs.shape[0], -1)
            x = self.image_conv(obs[:, :, :, :].reshape(obs.shape[0], -1))
        else:
            x = self.image_conv(obs.image[:, :, :, :].reshape(obs.image.shape[0], -1))
            #x = obs.image.reshape(obs.image.shape[0], -1)
        # x = self.image_conv(obs[:, 4, 2, 2].unsqueeze(dim=1))

        return x


class Image3dEncoder(nn.Module):
    def __init__(self, obs_space):
        super().__init__()

        self.embedding_size = 128 # image_embedding size

        self.image_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=9, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512, self.embedding_size)
        )

    def forward(self, obs):
        x = self.image_conv(obs)
        return x


class HistoryEncoder(nn.Module):
    def __init__(self, obs_space, use_cnn):
        super().__init__()

        # self.output_size = output_size
        self.use_cnn = use_cnn
        if use_cnn:
            self.image_conv = Image3dEncoder(obs_space)
        else:
            self.image_conv = ImageEncoder(obs_space)

        self.image_embedding_size = self.image_conv.embedding_size
        self.memory_rnn1 = nn.GRUCell(self.image_embedding_size, self.semi_memory_size)
        self.memory_rnn2 = nn.GRUCell(self.semi_memory_size, self.semi_memory_size)
        self.memory_rnn3 = nn.GRUCell(self.semi_memory_size, self.semi_memory_size)

        self.embedding_size = self.semi_memory_size

        self.predictor = nn.Sequential(
            nn.Linear(self.embedding_size + self.image_embedding_size, 256),
            nn.ReLU(),
            nn.Linear(256, self.embedding_size)
        )

    @property
    def memory_size(self):
        return 3*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return 256

    def forward(self, obs, memory):
        x = self.image_conv(obs)
        memory1, memory2, memory3 = memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:2*self.semi_memory_size], memory[:, 2*self.semi_memory_size:3*self.semi_memory_size]

        memory1 = self.memory_rnn1(x, memory1)
        memory2 = self.memory_rnn2(F.relu(memory1), memory2)
        memory3 = self.memory_rnn3(F.relu(memory2), memory3)

        memory = torch.cat([memory1, memory2, memory3], dim=1)

        prediction = self.predictor(torch.cat((memory1 + memory2, x), dim=-1))
        return prediction, memory



class BeliefVAEModel(nn.Module):
    def __init__(self, obs_space, x_dim, x_size, latent_dim=8, predict_full_state=False, use_cnn=False):
        super().__init__()

        self.x_dim = x_dim
        self.x_size = x_size
        self.latent_dim = latent_dim
        self.predict_full_state = predict_full_state
        print(x_dim, "xdim")
        print(x_size, "num pixels")
        print("latent dim of VAE is", latent_dim)

        self.history_model = HistoryEncoder(obs_space, use_cnn)
        self.context_dim = self.history_model.semi_memory_size
        state_features_dim = x_size

        # Outputs a mean and variance for a diagonal Gaussian in latent space
        self.vae_encoder = nn.Sequential(
            nn.Linear(state_features_dim + self.context_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2*self.latent_dim)
        ).to(device)

        if predict_full_state:
            # Outputs logits for a Categorical distribution with `self.x_dim` possible values
            self.vae_decoder = nn.Sequential(
                nn.Linear(self.latent_dim + self.context_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 2 * self.x_dim)
            ).to(device)
        else:
            self.vae_decoder = nn.Sequential(
                nn.Linear(self.latent_dim + self.context_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 2 * self.x_dim)
            ).to(device)

    @property
    def memory_size(self):
        return self.history_model.memory_size

    @property
    def semi_memory_size(self):
        return self.history_model.semi_memory_size

    def forward(self, obs, memory):
        encoding, memory = self.history_model(obs, memory)
        return encoding, memory

    # Return the encoder's distribution over z (latents) given x (state) and context (history of observations)
    def encoder_dist(self, x, context):
        out = self.vae_encoder(torch.cat([x, context], dim=1))
        mean = out[:, :self.latent_dim]
        std = F.softplus(out[:, self.latent_dim:], threshold=1) + 1e-1  # TODO try threshold between 1, ... 20

        return mean, std

    def decoder_dist(self, z, context):
        out = self.vae_decoder(torch.cat([z.to(device), context.to(device)], dim=1))
        mean = out[:, self.x_dim:]
        std = F.softplus(out[:, : self.x_dim], threshold=1) + 1e-1 #  + 1e-1
        return mean, std

    def sample(self, context):
        assert(len(context.shape)==2)
        bs = context.shape[0]
        zs = torch.randn(bs * 30, self.latent_dim)
        mean, std = self.decoder_dist(zs, context.repeat_interleave(30, dim=0))
        samples = Normal(mean, std).sample().reshape(bs, 30, -1)

        return samples
