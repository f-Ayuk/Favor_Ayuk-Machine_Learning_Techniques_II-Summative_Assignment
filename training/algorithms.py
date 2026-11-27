"""
RL agents built with TorchRL.

All four agents share:
  * the same environment interface (Discrete(512) tap patterns, 41-dim obs)
  * the same MLP body (TorchRL `MLP`)
  * TorchRL loss modules, value estimators and replay buffers

Value based:      DQN        -> torchrl.objectives.DQNLoss (+ HardUpdate target net)
Policy gradient:  REINFORCE  -> torchrl.objectives.ReinforceLoss (Monte Carlo returns)
                  A2C        -> torchrl.objectives.A2CLoss + GAE
                  PPO        -> torchrl.objectives.ClipPPOLoss + GAE
"""

import random

import numpy as np
import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.data import Categorical, LazyTensorStorage, TensorDictReplayBuffer
from torchrl.modules import MLP, ProbabilisticActor, QValueActor, ValueOperator
from torchrl.objectives import A2CLoss, ClipPPOLoss, DQNLoss, HardUpdate, ReinforceLoss
from torchrl.objectives.value import GAE

OBS_DIM = 41
N_TAPS = 9
N_ACTIONS = 2 ** N_TAPS
DEVICE = torch.device("cpu")   # the networks are small; CPU is fastest here
torch.set_num_threads(1)       # avoids thread contention when running many small nets

# TAP_BITS[a, i] = 1 if action a opens tap i
TAP_BITS = ((torch.arange(N_ACTIONS)[:, None] >> torch.arange(N_TAPS)) & 1).float()


def _entropy_kw(loss_cls, value):
    """TorchRL renamed entropy_coef -> entropy_coeff in v0.11; support both."""
    import inspect
    params = inspect.signature(loss_cls.__init__).parameters
    return {"entropy_coeff" if "entropy_coeff" in params else "entropy_coef": value}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ActionHead(nn.Module):
    """Maps an observation to one score per Discrete(512) action.

    Every head predicts one score per tap (plus a shared bias) and builds
        score(a) = bias + sum_i bit_i(a) * tap_score_i
    For a policy this is a product of 9 independent ON/OFF decisions; for DQN
    it is an additive (per-tap) decomposition of Q. The heads differ in how
    the per-tap scores are produced:

    head="shared"   (default) one small network is applied to every plot,
                    using that plot's readings + the farm-wide readings.
                    The 9 plots share weights, so each season gives 9x the
                    experience for the same rule.
    head="factored" one MLP reads the full observation and outputs 9 scores.
    head="flat"     a plain 512-way output layer (no tap structure at all).
    """

    LOCAL = ("moisture", "health", "exposure", "tap_states")   # 4 per plot
    N_GLOBAL = 5                                                # obs[36:41]

    def __init__(self, hidden=256, depth=2, head="shared", tap_prior=-1.5):
        super().__init__()
        self.head = head
        if head == "shared":
            plot_in = len(self.LOCAL) + self.N_GLOBAL
            self.plot_net = MLP(in_features=plot_in, out_features=1,
                                num_cells=[hidden // 2] * depth, activation_class=nn.Tanh)
            self.bias_net = MLP(in_features=OBS_DIM, out_features=1,
                                num_cells=[hidden // 2], activation_class=nn.Tanh)
            last = self.plot_net[-1]
        else:
            out = N_TAPS + 1 if head == "factored" else N_ACTIONS
            self.body = MLP(in_features=OBS_DIM, out_features=out,
                            num_cells=[hidden] * depth, activation_class=nn.Tanh)
            last = self.body[-1]
        # Start with taps mostly OFF (p(ON) = sigmoid(-1.5) ~ 0.18). Random
        # 50/50 tap patterns flood the farm and empty the reservoir.
        if tap_prior is not None and head != "flat":
            with torch.no_grad():
                last.bias.fill_(tap_prior)
        self.register_buffer("bits", TAP_BITS)

    def tap_scores(self, obs):
        if self.head == "shared":
            local = obs[..., :36].unflatten(-1, (len(self.LOCAL), N_TAPS)).transpose(-1, -2)
            glob = obs[..., 36:].unsqueeze(-2).expand(*local.shape[:-1], self.N_GLOBAL)
            taps = self.plot_net(torch.cat([local, glob], -1)).squeeze(-1)
            return taps, self.bias_net(obs)
        out = self.body(obs)
        return out[..., :N_TAPS], out[..., N_TAPS:]

    def forward(self, obs):
        if self.head == "flat":
            return self.body(obs)
        taps, bias = self.tap_scores(obs)
        return taps @ self.bits.T + bias


def _value_net(hidden, depth=2):
    return MLP(in_features=OBS_DIM, out_features=1,
               num_cells=[hidden] * depth, activation_class=nn.Tanh)


def obs_td(obs):
    return TensorDict({"observation": torch.as_tensor(obs, dtype=torch.float32)}, batch_size=[])


def transition_td(obs, action, reward, next_obs, terminated, truncated, log_prob=None):
    """One environment step in the TensorDict layout TorchRL losses expect."""
    td = TensorDict({
        "observation": torch.as_tensor(obs, dtype=torch.float32),
        "action": torch.tensor(action, dtype=torch.long),
        "next": TensorDict({
            "observation": torch.as_tensor(next_obs, dtype=torch.float32),
            "reward": torch.tensor([reward], dtype=torch.float32),
            "terminated": torch.tensor([terminated]),
            "truncated": torch.tensor([truncated]),
            "done": torch.tensor([terminated or truncated]),
        }, batch_size=[]),
    }, batch_size=[])
    if log_prob is not None:
        td["sample_log_prob"] = torch.tensor(log_prob, dtype=torch.float32)
    return td


# ===========================================================================
# DQN
# ===========================================================================
class DQNAgent:
    name = "DQN"

    def __init__(self, lr=1e-3, gamma=0.99, batch_size=64, buffer_size=50_000,
                 target_update=500, eps_start=1.0, eps_end=0.05,
                 eps_decay_steps=20_000, hidden=256, head="shared", double_dqn=True,
                 learning_starts=1_000, reward_scale=0.1, explore_p_on=0.2, **_):
        self.gamma = gamma
        self.explore_p_on = explore_p_on
        self.batch_size = batch_size
        self.eps_start, self.eps_end = eps_start, eps_end
        self.eps_decay_steps = eps_decay_steps
        self.learning_starts = learning_starts
        self.reward_scale = reward_scale

        self.actor = QValueActor(ActionHead(hidden, head=head), in_keys=["observation"],
                                 spec=Categorical(N_ACTIONS))
        self.loss = DQNLoss(self.actor, action_space="categorical", delay_value=True,
                            double_dqn=double_dqn, loss_function="smooth_l1")
        self.loss.make_value_estimator(gamma=gamma)
        self.target_updater = HardUpdate(self.loss, value_network_update_interval=target_update)
        self.optimizer = torch.optim.Adam(self.loss.parameters(), lr=lr)
        self.buffer = TensorDictReplayBuffer(storage=LazyTensorStorage(buffer_size),
                                             batch_size=batch_size)
        self.steps = 0

    @property
    def epsilon(self):
        frac = min(1.0, self.steps / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    @torch.no_grad()
    def act(self, obs, greedy=False):
        if not greedy and random.random() < self.epsilon:
            # Explore with a random tap pattern where each tap is ON with p=0.2
            taps = np.random.random(N_TAPS) < self.explore_p_on
            return int(np.sum(taps << np.arange(N_TAPS)))
        return int(self.actor(obs_td(obs))["action"])

    def observe(self, obs, action, reward, next_obs, terminated, truncated):
        """Store one transition and do one gradient step."""
        self.buffer.add(transition_td(obs, action, reward * self.reward_scale,
                                      next_obs, terminated, truncated))
        self.steps += 1
        if len(self.buffer) < self.learning_starts:
            return None
        batch = self.buffer.sample()
        loss = self.loss(batch)["loss"]
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.loss.parameters(), 10.0)
        self.optimizer.step()
        self.target_updater.step()
        return float(loss)

    def state_dict(self):
        return {"algorithm": "DQN", "actor": self.actor.state_dict()}

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd["actor"])


# ===========================================================================
# Policy gradient family (REINFORCE, A2C, PPO)
# ===========================================================================
class PolicyGradientAgent:
    """One class, three update rules. Each episode batch is collected with the
    current stochastic policy, then `update(episodes)` is called."""

    def __init__(self, algorithm, lr=3e-4, gamma=0.99, hidden=256,
                 entropy_coef=0.01, critic_coef=0.5, gae_lambda=0.95,
                 clip_epsilon=0.2, ppo_epochs=4, minibatch_size=256,
                 episodes_per_batch=4, use_baseline=True, reward_scale=0.1,
                 max_grad_norm=0.5, head="shared", **_):
        self.name = algorithm.upper()
        assert self.name in {"REINFORCE", "A2C", "PPO"}
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs
        self.minibatch_size = minibatch_size
        self.episodes_per_batch = episodes_per_batch
        self.use_baseline = use_baseline
        self.reward_scale = reward_scale
        self.max_grad_norm = max_grad_norm

        self.policy_net = ActionHead(hidden, head=head)
        logits = TensorDictModule(self.policy_net, in_keys=["observation"],
                                  out_keys=["logits"])
        self.actor = ProbabilisticActor(logits, in_keys=["logits"],
                                        distribution_class=torch.distributions.Categorical,
                                        return_log_prob=True)
        self.critic = ValueOperator(_value_net(hidden), in_keys=["observation"])

        if self.name == "PPO":
            self.loss = ClipPPOLoss(self.actor, self.critic, clip_epsilon=clip_epsilon,
                                    **_entropy_kw(ClipPPOLoss, entropy_coef),
                                    critic_coef=critic_coef,
                                    loss_critic_type="smooth_l1", normalize_advantage=True)
        elif self.name == "A2C":
            self.loss = A2CLoss(self.actor, self.critic,
                                **_entropy_kw(A2CLoss, entropy_coef),
                                critic_coef=critic_coef, loss_critic_type="smooth_l1")
        else:
            # TorchRL's REINFORCE trains a value net alongside the policy; it is
            # only used as a baseline when use_baseline=True.
            self.loss = ReinforceLoss(self.actor, self.critic, loss_critic_type="smooth_l1")

        self.gae = GAE(gamma=gamma, lmbda=gae_lambda, value_network=self.critic,
                       average_gae=False)
        self.optimizer = torch.optim.Adam(self.loss.parameters(), lr=lr)

    @torch.no_grad()
    def act(self, obs, greedy=False):
        # Same distribution as self.actor, called directly for speed.
        logits = self.policy_net(torch.as_tensor(obs, dtype=torch.float32))
        if greedy:
            return int(logits.argmax()), None
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return int(action), float(dist.log_prob(action))

    # ------------------------------------------------------------------
    def _prepare(self, episode):
        """Add advantage / value_target to one episode TensorDict [T]."""
        episode = episode.clone()
        episode["next", "reward"] = episode["next", "reward"] * self.reward_scale
        if self.name in {"A2C", "PPO"}:
            with torch.no_grad():
                self.gae(episode)
        else:  # REINFORCE: Monte Carlo discounted return G_t
            rewards = episode["next", "reward"].squeeze(-1)
            returns = torch.zeros_like(rewards)
            running = 0.0
            for t in reversed(range(len(rewards))):
                running = rewards[t] + self.gamma * running
                returns[t] = running
            returns = returns.unsqueeze(-1)
            episode["value_target"] = returns
            if self.use_baseline:
                with torch.no_grad():
                    baseline = self.critic(episode.select("observation").clone())["state_value"]
                episode["advantage"] = returns - baseline
            else:
                episode["advantage"] = returns
        return episode

    def _step(self, batch):
        losses = self.loss(batch)
        loss = sum(v for k, v in losses.items() if k.startswith("loss_"))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.loss.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return float(loss)

    def update(self, episodes):
        """Returns (loss, policy entropy) for logging."""
        data = torch.cat([self._prepare(ep) for ep in episodes], dim=0)
        if self.name != "PPO":
            adv = data["advantage"]
            data["advantage"] = (adv - adv.mean()) / (adv.std() + 1e-8)
            loss = self._step(data)           # one on-policy gradient step
        else:
            # PPO: several epochs of clipped updates over shuffled minibatches
            n = data.shape[0]
            for _ in range(self.ppo_epochs):
                perm = torch.randperm(n)
                for start in range(0, n, self.minibatch_size):
                    loss = self._step(data[perm[start:start + self.minibatch_size]])
        with torch.no_grad():
            logits = self.policy_net(data["observation"])
            entropy = torch.distributions.Categorical(logits=logits).entropy().mean()
        return loss, float(entropy)

    def state_dict(self):
        return {"algorithm": self.name, "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict()}

    def load_state_dict(self, sd):
        self.actor.load_state_dict(sd["actor"])
        self.critic.load_state_dict(sd["critic"])


def make_agent(algorithm, config):
    algorithm = algorithm.upper()
    if algorithm == "DQN":
        return DQNAgent(**config)
    return PolicyGradientAgent(algorithm, **config)


def save_agent(agent, config, path):
    sd = agent.state_dict()
    sd["config"] = config
    torch.save(sd, path)


def load_agent(path):
    sd = torch.load(path, map_location=DEVICE, weights_only=False)
    agent = make_agent(sd["algorithm"], sd.get("config", {}))
    agent.load_state_dict(sd)
    return agent
