"""
This file defines 3 modules:
- Agent: module for DDPG agent
- OUNoise: module for Ornstein–Uhlenbeck noise
- ReplayBuffer: module for experience replay buffer

"""

# import libraries
import numpy as np
import random
import copy
import torch
from collections import namedtuple, deque
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

# import the modules for the actor and critic
from model import Actor, Critic

# hyper parameters
BUFFER_SIZE = 100000 # the size of the replay buffer
BATCH_SIZE = 256 # batch sise
GAMMA = 0.99  #0.99 # discount factor
TAU = 0.001 # a parameter for softupdate
LR_ACTOR = 0.002 # learning rate for actor model
LR_CRITIC = 0.001 # learning rate for critic model

UPDATE_EVERY = 20 # the learning process is done every UPDATE_EVERY time steps
WEIGHT_DECAY_ACTOR = 0.0 # weight decay for the optimizer of the actor
WEIGHT_DECAY_CRITIC = 0.0 # weight decay for the optimizer of the critic

# In case one wants to decrese the size of the noise, tune below.
# No decay for the current parametrization.
EPSILON = 1.0 # initial size factor of the noise
EPSILON_DECAY = 1.0 # a parameter for the decay of the size factor of the noise

# choose device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # use this in case one wants to work with cpu (even when gpu is installed)
# device = torch.device("cpu")

class Agent():
    """
    This module defines DDPG agent.

    """

    def __init__(self, state_size, action_size, seed = 123):
        """
        Initialization.

        (input)
        - state_size (int): size of a state (=33)
        - action_size (int): size of an action (=4)
        - seed (int): random seed

        """

        self.seed = random.seed(seed) # random seed
        self.action_size = action_size # action size
        self.state_size = state_size # state size

        # neural networks (local and target) for actor
        self.actor_local = Actor(self.state_size, self.action_size, seed).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr = LR_ACTOR, weight_decay = WEIGHT_DECAY_ACTOR)

        # neural networks (local and target) for critic
        self.critic_local = Critic(self.state_size, self.action_size, seed).to(device)
        self.critic_target = Critic(self.state_size, self.action_size, seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr = LR_CRITIC, weight_decay = WEIGHT_DECAY_CRITIC)

        # set the weights of the target networks to be the same as those of the local networks
        self.soft_update(self.critic_local, self.critic_target, 1.0)
        self.soft_update(self.actor_local, self.actor_target, 1.0)

        # replay buffer
        self.buffer  = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)

        # elapsed time in a given episode (modulo UPDATE_EVERY)
        self.t_step = 0

        # noise for action
        self.noise = OUNoise(self.action_size, seed)

        # epsilon used for decaying of the size of the noise
        self.epsilon = EPSILON


    def step(self, states, actions, rewards, next_states, dones):
        """
        This method adds a tuple of (state, action, reward, next_state, done) to
        the replay buffer and trains the actor and critic models.

        (input)
        - states (float, size:(20, 33)): states for all the 20 agents
        - actions (float, size:(20, 4)): actions for all the agents
        - rewards (float, size:(20, 1)): rewards for all the agents
        - next_states (float, size:(20, 33)): next_states for all the agents
        - dones (bool, size:(20, 1)): if an episode is done or not for all the agents

        """

        # add the tuple of state etc. from all the agents to the replay buffer
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.buffer.add(state, action, reward, next_state, done)

        # every UPDATE_EVERY time steps, if the buffer size is bigger than
        # the batch size, do the learning process for 10 times
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            if len(self.buffer) > BATCH_SIZE:
                for _ in range(10):
                    # sample experiences in the replay buffer
                    experiences = self.buffer.sample()
                    # learnig process
                    self.learn(experiences, GAMMA)
                    # update the size factor of the noise if needed
                    self.epsilon *= EPSILON_DECAY


    def act(self, state, add_noise = True):
        """
        This method is used for selecting a next action based
        on the actor model (and add noise).

        """

        # convert the state vector to a tensor
        state = torch.from_numpy(state).float().to(device)

        # choose action based on the actor model.
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()

        # add noise
        if add_noise:
            action += self.noise.sample() * self.epsilon

        return np.clip(action, -1, 1) # clip such that each entry in the action is in [-1, 1]


    def reset(self):
        """
        This method is for resetting the noise and elapsed time (modulo UPDATE_EVERY).
        This method is used at the beginning of each episode.

        """

        self.noise.reset()
        self.t_step  = 0


    def learn(self, experiences, gamma):
        """
        This method is for updating the weights for the actor-critic model
        by using DDPG algorithm.

        (input)
        - experience: tuple of states, actions, rewards, next_states and dones
        - gamma (float): discount factor

        """

        # read states etc. from the experience tuple
        states, actions, rewards, next_states, dones = experiences

        # update the local critic model
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + gamma * Q_targets_next * (1 - dones)
        Q_expected = self.critic_local(states, actions)

        # loss and optmizer for the critic model and back propagation
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1.0) # gradient clipping
        self.critic_optimizer.step()

        # update the local actor model
        actions_pred = self.actor_local(states)
        # loss and optmizer for the critic model and back propagation
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # soft-update of the target networks
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)


    def soft_update(self, local_model, target_model, tau):
        """
        This method is for the soft-update of the weights of the target neural networks.

        """

        for l_param, t_param in zip(local_model.parameters(), target_model.parameters()):
            t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)



class OUNoise():
    """
    This module is for generating noise with OU process.

    """

    def __init__(self, size, seed, mu = 0.0, theta = 0.15, sigma = 0.1):
        """
        Initialization.

        (input)
        - size (int): dimension of the noise
        - seed (int): random manual_seed
        - mu (float): mu parameter for OU noise
        - theta (float): theta parameter of OU noise
        - sigma (float): sigma parameter for OU noise

        """

        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)

        # reset seed of noise
        self.reset()


    def reset(self):
        """
        For reset the seed for noise.
        """

        self.state = copy.copy(self.mu)


    def sample(self):
        """
        For sampling noises with OU process.

        """

        x = self.state
        # OU process
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx

        return self.state



class ReplayBuffer():
    """
    This module is for creating replay buffer and
    sampling experience tuples from it.

    """

    def __init__(self, buffer_size, batch_size, seed = 31):
        """
        For initialization.

        (input)
        - buffer_size (int): buffer size
        - batch_size (int): batch size for sampling
        - seed (int): random seed
        """

        self.seed = random.seed(seed) # random seed
        self.buffer_size = buffer_size # buffer size
        self.batch_size = batch_size # batch size
        self.memory = deque(maxlen = self.buffer_size) # memory for storing experiences
        field_names = ['state', 'action', 'reward', 'next_state', 'done'] # names for experience tuple
        self.experience = namedtuple("Experience", field_names = field_names) # experience as a named tuple


    def add(self, state, action, reward, next_state, done):
        """
        This method is for adding an experience to the replay buffer.

        """

        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)


    def sample(self):
        """
        This method is for sampling batch_size of experience tuples
        from the replay buffer.

        """

        # sample from the replay buffer.
        experiences = random.sample(self.memory, k = self.batch_size)

        # reformat to tensors so that we can use with the neural networks.
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """
        This returns the size of the replay buffer.

        """

        return len(self.memory)
