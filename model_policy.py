import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import torch_ac


# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self,  obs_space,
                        action_space,
                        aux_info_embedding_size,
                        algo_name,
                        use_memory=False,
                        use_text=False,
                        aux_info=False,
                        latent_dim_f=16,
                        use_cnn=False):
        """
        :param obs_space:
        :param action_space:
        :param aux_info_embedding_size: The dimensions of the auxiliary info z vector
        :param algo_name: Name of the algorithm
        :param use_memory: Whether this policy uses memory or is memoryless
        :param use_text: Whether this policy conditions on text
        :param aux_info: Whether this policy uses auxiliary information
        """
        super().__init__()

        # Decide which components are enabled
        self.aux_info = aux_info
        self.use_text = use_text
        self.use_memory = use_memory
        self.aux_info_embedding_size = aux_info_embedding_size
        self.algo_name = algo_name
        self.latent_dim_f = latent_dim_f
        self.use_cnn = use_cnn
        self.action_space = action_space

        self.embedding_size = 64

        # Define image embedding
        if use_cnn:
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

        else:
            n = obs_space["image"][0]
            self.width = n
            m = obs_space["image"][1]
            self.height = m
            k = obs_space["image"][2]
            input_size = n * m * k

            # Define image embedding
            self.image_conv = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_size, 64),
                nn.ReLU(),
                nn.Linear(64, self.embedding_size))

        # Resize image embedding

        if self.aux_info:
            self.embedding_size += self.aux_info_embedding_size
        if self.algo_name == "belief_vae":
            self.belief_encoder = nn.Sequential(
                nn.Linear(self.latent_dim_f, 64),
                nn.ReLU(),
                nn.Linear(64, 64)
            )

            self.belief_agg = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, 64)
            )

            self.embedding_size += 64

        if self.use_cnn:
            action_space_dim = 4
        else:
            action_space_dim = action_space.n

        # Define actor's model
        if use_cnn:
            self.actor = nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, action_space_dim)
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, action_space_dim)
            )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2 * self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.embedding_size


    def encode_image(self, obs):
        if self.use_cnn:
            x = self.image_conv(obs)  # B x C X H X W
        else:
            # image 10 x 10 x 5 (item idx, colour, status code, boolean that describes if agent is here, agent direction facing)
            x = self.image_conv(obs)  # B x C X H X W
        return x



    def forward(self, obs, memory, belief_samples):
        x = self.encode_image(obs)

        if self.algo_name == "belief_vae":
            belief_enc = self.belief_agg(
                torch.mean(self.belief_encoder(belief_samples), dim=1)
            )

            embedding = torch.cat((x, belief_enc), dim=1)

        x = self.actor(embedding)

        if self.use_cnn:
            dist = Normal(loc=2*(F.sigmoid(x[:, :self.action_space.shape[0]]) - 0.5),
                          scale=F.sigmoid(x[:, self.action_space.shape[0]:]) + 1e-3)
        else:
            dist = Categorical(logits=F.log_softmax(x, dim=1))

        c = self.critic(embedding)
        value = c.squeeze(1)

        return dist, value, memory, 0


    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]


class VanillaACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self,  obs_space,
                        action_space,
                        use_memory=False,
                        use_cnn=False):
        """
        :param obs_space:
        :param action_space:
        :param aux_info_embedding_size: The dimensions of the auxiliary info z vector
        :param algo_name: Name of the algorithm
        :param use_memory: Whether this policy uses memory or is memoryless
        :param use_text: Whether this policy conditions on text
        :param aux_info: Whether this policy uses auxiliary information
        """
        super().__init__()

        # Decide which components are enabled
        self.use_memory = use_memory
        self.use_cnn = use_cnn
        self.obs_space = obs_space
        self.action_space = action_space

        # Define image embedding

        self.embedding_size = self.semi_memory_size
        self.image_embedding_size = 256

        # Define image embedding
        if self.use_cnn:
            if self.use_memory:
                cnn_output_size = self.image_embedding_size
            else:
                cnn_output_size = self.semi_memory_size
            
            self.image_conv = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=8, kernel_size=9, stride=4),
                nn.ReLU(),
                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(512, cnn_output_size)
            )

        else:
            input_size = obs_space['image'][0] * obs_space['image'][1] * obs_space['image'][2]
            if self.use_memory:
                self.image_conv = nn.Linear(input_size, self.image_embedding_size)
            else:
                self.image_conv = nn.Linear(input_size, self.semi_memory_size)
        n = obs_space["image"][0]
        self.width = n
        m = obs_space["image"][1]
        self.height = m
        
        # Define memory
        if self.use_memory:
            self.memory_rnn1 = nn.GRUCell(self.image_embedding_size, self.semi_memory_size)
            self.memory_rnn2 = nn.GRUCell(self.semi_memory_size, self.semi_memory_size)
            self.memory_rnn3 = nn.GRUCell(self.semi_memory_size, self.semi_memory_size)
            self.predictor = nn.Sequential(
                nn.Linear(self.embedding_size + self.image_embedding_size, 256),
                nn.ReLU(),
                nn.Linear(256, self.embedding_size)
            )

        if self.use_cnn:
            action_space_dim = 4
        else:
            action_space_dim = action_space.n

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_space_dim)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 3*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return 256

    def forward(self, obs, memory):
        x = obs.image
        if not self.use_cnn:
            x = x.reshape(x.shape[0], -1)
        x = self.image_conv(x)  # B x C X H X W

        if self.use_memory:
            memory1, memory2, memory3 = (memory[:, :self.semi_memory_size],
                                         memory[:, self.semi_memory_size:2*self.semi_memory_size],
                                         memory[:, 2*self.semi_memory_size:3*self.semi_memory_size]
                                         )

            memory1 = self.memory_rnn1(x, memory1)
            memory2 = self.memory_rnn2(F.relu(memory1), memory2)
            memory3 = self.memory_rnn3(F.relu(memory2), memory3)

            memory = torch.cat([memory1, memory2, memory3], dim=1)

            embedding = self.predictor(torch.cat((memory1 + memory2 + memory3, x), dim=-1))

        else:
            embedding = x

        x = self.actor(embedding)
        if self.use_cnn:
            dist = Normal(loc=2*(F.sigmoid(x[:, :self.action_space.shape[0]]) - 0.5),
                          scale=F.sigmoid(x[:, self.action_space.shape[0]:]) + 1e-3)
        else:
            dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

