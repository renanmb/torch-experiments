# logger.warn(
# [skrl:INFO] Environment wrapper: Isaac Lab (single-agent)
# [skrl:INFO] Seed: 42
# ==================================================
# Shared model (roles): ['policy', 'value']
# ==================================================

# --------------------------------------------------
# ==================================================
# Observations for Agent_id agent Shared model (roles): ['policy', 'value']
# ==================================================

# Box(-inf, inf, (100, 100, 3), float32)
# --------------------------------------------------

# this is for the Environment Isaac-Cartpole-RGB-Camera-Direct-v0
# isaaclab -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Cartpole-RGB-Camera-Direct-v0 --headless --enable_cameras
# isaaclab -p scripts/reinforcement_learning/skrl/play.py --task Isaac-Cartpole-RGB-Camera-Direct-v0 --num_envs 32 --enable_cameras

class SharedModel(GaussianMixin,DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=False,
            clip_log_std=True,
            min_log_std=-20.0,
            max_log_std=2.0,
            reduction="sum",
            role="policy",
        )
        DeterministicMixin.__init__(self, clip_actions=False, role="value")

        self.features_extractor_container = nn.Sequential(
            nn.LazyConv2d(out_channels=32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.net_container = nn.Sequential(
            nn.LazyLinear(out_features=512),
            nn.ELU(),
        )
        self.policy_layer = nn.LazyLinear(out_features=self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.full(size=(self.num_actions,), fill_value=0.0), requires_grad=True)
        self.value_layer = nn.LazyLinear(out_features=1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)
    
    def compute(self, inputs, role=""):
        if role == "policy":
            states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
            taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
            features_extractor = self.features_extractor_container(torch.permute(states, (0, 3, 1, 2)))
            net = self.net_container(features_extractor)
            self._shared_output = net
            output = self.policy_layer(net)
            return output, self.log_std_parameter, {}
        elif role == "value":
            if self._shared_output is None:
                states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
                taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
                features_extractor = self.features_extractor_container(torch.permute(states, (0, 3, 1, 2)))
                net = self.net_container(features_extractor)
                shared_output = net
            else:
                shared_output = self._shared_output
            self._shared_output = None
            output = self.value_layer(shared_output)
            return output, {}