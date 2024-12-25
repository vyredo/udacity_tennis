from ActorCritic import Actor, Critic
import numpy as np
import torch
from torch import optim, nn
import random
from OUNoise import OUNoise
from ReplayBuffer import ReplayBuffer
from ActorCritic import Actor, Critic
import numpy as np
import random
import torch
from torch import optim, nn
from OUNoise import OUNoise
from ReplayBuffer import ReplayBuffer


class DDPGAgent:
    def __init__(self, state_size, action_size, random_seed, device,
                 tau,
                 BATCH_SIZE, gamma, learning_rate, buffer_size, num_agents=2):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.device = device
        self.batch_size = BATCH_SIZE
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.num_agents = num_agents

        # Actor Network
        self.actor_local = Actor(
            state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(
            state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=self.learning_rate)
        self.actor_target.load_state_dict(self.actor_local.state_dict())

        # Critic Network
        self.critic_local = Critic(
            state_size, action_size, random_seed, num_agents=num_agents).to(device)
        self.critic_target = Critic(
            state_size, action_size, random_seed, num_agents=num_agents).to(device)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(), lr=self.learning_rate * 3, weight_decay=0)
        self.critic_target.load_state_dict(self.critic_local.state_dict())

        self.memory = ReplayBuffer(
            action_size * num_agents, self.buffer_size, BATCH_SIZE, random_seed)
        self.noise = OUNoise(action_size, random_seed)
        self.tau = tau

    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def step(self, states, actions, rewards, next_states, dones):
        self.memory.add(states, actions, rewards, next_states, dones)
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, gamma=self.gamma)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                tau * local_param.data + (1.0 - tau) * target_param.data)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Move tensors to the gpu
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Split next_states into individual agent states
        next_states_agent1 = next_states[:, :self.state_size]
        next_states_agent2 = next_states[:, self.state_size:]

        # Predict next actions for both agents
        actions_next_agent1 = self.actor_target(next_states_agent1)
        actions_next_agent2 = self.actor_target(next_states_agent2)
        actions_next = torch.cat(
            [actions_next_agent1, actions_next_agent2], dim=1)

        # Compute Q targets
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)

        # Compute critic loss
        critic_loss = nn.MSELoss()(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        actions_pred_agent1 = self.actor_local(states[:, :self.state_size])
        actions_pred_agent2 = self.actor_local(states[:, self.state_size:])
        actions_pred = torch.cat(
            [actions_pred_agent1, actions_pred_agent2], dim=1)

        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft Update of Target Networks
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)
