import numpy as np
import random
import copy
from collections import namedtuple, deque

from model import Actor, Critic

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(5e4)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
LEARNPERSTEP = 3        # LEARNPERSTEP define the amount of learning performed per agent step.
RANDOM_ACT_TILL = 300   # Take random action till RANDOM_ACT_TILL episodes
NOISE_DECAY =  0.0005   # Per episode noise decay factor

# as multiple agents simultaneouly add data to reply buffer we can train the network multiple times

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MultiAgents():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, n_agents, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.n_agents = n_agents
        self.seed = random.seed(random_seed)

        self.ma = [Agent(state_size, action_size, i, n_agents, random_seed) for i in range(n_agents)]

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)


    def step(self, states, actions, rewards, next_states, dones,i_episode=0):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.memory.add(states, actions, rewards, next_states, dones)

        if len(self.memory) > BATCH_SIZE and i_episode > RANDOM_ACT_TILL:
            for _ in range(LEARNPERSTEP):
                for agent in self.ma:
                    experiences = self.memory.sample()
                    self.learn(experiences, agent, GAMMA)

                for agent in self.ma:
                    agent.soft_update(agent.critic_local,
                          agent.critic_target,
                          TAU)
                    agent.soft_update(agent.actor_local,
                          agent.actor_target,
                          TAU)

    def learn(self, experiences, agent, gamma):
        states, actions, _, _, _ = experiences

        actions_target =[agent_j.actor_target(states.index_select(1, torch.tensor([j]).to(device)).squeeze(1)) for j, agent_j in enumerate(self.ma)]

        agent_action_pred = agent.actor_local(states.index_select(1, agent.agent_idx).squeeze(1))
        actions_pred = [agent_action_pred if j==agent.agent_idx.numpy()[0] else actions.index_select(1, torch.tensor([j]).to(device)).squeeze(1) for j, agent_j in enumerate(self.ma)]

        agent.learn(experiences,
                    gamma,
                    actions_target,
                    actions_pred)


    def act(self, states, i_episode=0, add_noise=True):
        if i_episode > RANDOM_ACT_TILL:
            actions = [np.squeeze(agent.act(np.expand_dims(state, axis=0), i_episode, add_noise), axis=0) for agent, state in zip(self.ma, states)]
            return_actions = np.stack(actions)
        else:
            actions = np.random.randn(self.n_agents, self.action_size) # select an action (for each agent)
            return_actions = np.clip(actions, -1, 1)
        return return_actions


    def reset(self):
        for agent in self.ma:
            agent.reset()


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, agent_idx ,n_agents, random_seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.agent_idx = torch.tensor([agent_idx]).to(device)
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        # For multi-agent DDPG the critic network is working on Set of environment state from all agents
        # For multi-agent DDPG the critic network is working on Set of actions from all agents
        self.critic_local = Critic(n_agents*state_size, n_agents*action_size, random_seed).to(device)
        self.critic_target = Critic(n_agents*state_size, n_agents*action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        self.soft_update(self.critic_local, self.critic_target, 1)
        self.soft_update(self.actor_local, self.actor_target, 1)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)

    def act(self, state, i_episode=0, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise and i_episode > RANDOM_ACT_TILL:
            # decay noise after the RANDOM_ACT_TILL episodes
            action += (1 - min(1,(i_episode - RANDOM_ACT_TILL) * NOISE_DECAY)) * self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma, actions_target, actions_pred):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        rewards = rewards.unsqueeze(-1)
        dones = dones.unsqueeze(-1)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        # actions_next = self.actor_target(next_states)
        # For MA-DDPG combine the actions
        actions_target = torch.cat(actions_target, dim=1).to(device)

        #Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets_next = self.critic_target(next_states.reshape(next_states.shape[0], -1), actions_target.reshape(next_states.shape[0], -1))

        # Compute Q targets for current states (y_i)
        #Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_targets = rewards.index_select(1, self.agent_idx).squeeze(1) + (gamma * Q_targets_next * (1 - dones.index_select(1, self.agent_idx).squeeze(1)))

        # Compute critic loss
        # Q_expected = self.critic_local(states, actions)
        Q_expected = self.critic_local(states.reshape(states.shape[0], -1), actions.reshape(actions.shape[0], -1))

        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        #actions_pred = self.actor_local(states)
        actions_pred = torch.cat(actions_pred, dim=1).to(device)

        #actor_loss = -self.critic_local(states, actions_pred).mean()
        actor_loss = -self.critic_local(states.reshape(states.shape[0], -1), actions_pred.reshape(actions_pred.shape[0], -1)).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        #self.soft_update(self.critic_local, self.critic_target, TAU)
        #self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state



class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.stack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.stack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.stack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
