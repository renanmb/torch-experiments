# this is for the SKRL MAPPO multi agent stuff
# --task Isaac-Cart-Double-Pendulum-Direct-v0 --algorithm MAPPO 
# Train
# isaaclab -p scripts/reinforcement_learning/skrl/train.py --task Isaac-Cart-Double-Pendulum-Direct-v0 --algorithm MAPPO --headless
# Play
# isaaclab -p scripts/reinforcement_learning/skrl/play.py --task Isaac-Cart-Double-Pendulum-Direct-v0 --algorithm MAPPO --num_envs 32

# [2025-05-28 13:30:06,438][skrl][INFO] - Environment wrapper: Isaac Lab (multi-agent)
# [skrl:INFO] Seed: 42
# [2025-05-28 13:30:06,438][skrl][INFO] - Seed: 42
# ==================================================
# Model (role): policy
# ==================================================

class GaussianModel(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions,
                    clip_log_std, min_log_std, max_log_std, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net_container = nn.Sequential(
            nn.LazyLinear(out_features=32),
            nn.ELU(),
            nn.LazyLinear(out_features=32),
            nn.ELU(),
            nn.LazyLinear(out_features=self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.full(size=(self.num_actions,), fill_value=0.0), requires_grad=True)

    def compute(self, inputs, role=""):
        states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
        taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
        output = self.net_container(states)
        return output, self.log_std_parameter, {}
    
# --------------------------------------------------
# ==================================================
# Model (role): value
# ==================================================

class DeterministicModel(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net_container = nn.Sequential(
            nn.LazyLinear(out_features=32),
            nn.ELU(),
            nn.LazyLinear(out_features=32),
            nn.ELU(),
            nn.LazyLinear(out_features=1),
        )

    def compute(self, inputs, role=""):
        states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
        taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
        output = self.net_container(states)
        return output, {}
    
# --------------------------------------------------
# ==================================================
# Model (role): policy
# ==================================================

class GaussianModel(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions,
                    clip_log_std, min_log_std, max_log_std, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.net_container = nn.Sequential(
            nn.LazyLinear(out_features=32),
            nn.ELU(),
            nn.LazyLinear(out_features=32),
            nn.ELU(),
            nn.LazyLinear(out_features=self.num_actions),
        )
        self.log_std_parameter = nn.Parameter(torch.full(size=(self.num_actions,), fill_value=0.0), requires_grad=True)

    def compute(self, inputs, role=""):
        states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
        taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
        output = self.net_container(states)
        return output, self.log_std_parameter, {}
    
# --------------------------------------------------
# ==================================================
# Model (role): value
# ==================================================

class DeterministicModel(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net_container = nn.Sequential(
            nn.LazyLinear(out_features=32),
            nn.ELU(),
            nn.LazyLinear(out_features=32),
            nn.ELU(),
            nn.LazyLinear(out_features=1),
        )

    def compute(self, inputs, role=""):
        states = unflatten_tensorized_space(self.observation_space, inputs.get("states"))
        taken_actions = unflatten_tensorized_space(self.action_space, inputs.get("taken_actions"))
        output = self.net_container(states)
        return output, {}
    
# --------------------------------------------------
# ==================================================
# Observation for Agent_id cart --- Model (role): policy
# ==================================================

# Box(-inf, inf, (4,), float32)
# --------------------------------------------------
# ==================================================
# Observation for Agent_id cart --- Model (role): value
# ==================================================

# Box(-inf, inf, (7,), float32)
# --------------------------------------------------
# ==================================================
# Observation for Agent_id pendulum --- Model (role): policy
# ==================================================

# Box(-inf, inf, (3,), float32)
# --------------------------------------------------
# ==================================================
# Observation for Agent_id pendulum --- Model (role): value
# ==================================================

# Box(-inf, inf, (7,), float32)
# --------------------------------------------------
# --------------------------------------------------
# ERROR 01 
# --------------------------------------------------

# Need to make a look to get all models or agent_id

# Error executing job with overrides: []
# Traceback (most recent call last):
#   File "/home/goat/Documents/GitHub/renanmb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/utils/hydra.py", line 101, in hydra_main
#     func(env_cfg, agent_cfg, *args, **kwargs)
#   File "/home/goat/Documents/GitHub/renanmb/IsaacLab/scripts/reinforcement_learning/skrl/train.py", line 186, in main
#     runner = Runner(env, agent_cfg)
#   File "/home/goat/Documents/GitHub/renanmb/skrl/skrl/utils/runner/torch/runner.py", line 52, in __init__
#     num_obs = obs['agent']['policy']
# KeyError: 'agent'