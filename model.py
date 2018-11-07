"""
This file defines the modules for the neural networks
corresponding to the actor and critic for the DDPG algorithm.

"""

# import libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    """
    This function is used for initializing
    weights of a given layer such that they obey
    the uniform distribution with max = (1.0/(# of units))**0.5
    and min=-(1.0/(# of units))**0.5

    """
    fan_in = layer.weight.data.size()[0]
    lim = np.sqrt(1.0/fan_in)
    return (-lim, lim)


class Actor(nn.Module):
    """
    This module defines a neural network for the actor model.
    This neural network takes a state (size:33) as an input
    and returns an action (size:4).

    """

    def __init__(self, state_size, action_size, seed = 21):
        """
        Initialization.

        (input)
        - state_size (int): state size (=33)
        - action_size (float): action size (=4)
        - seed (int): random seed

        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed) # random seed
        self.state_size = state_size # state size
        self.action_size = action_size # action size
        hidden_units = [128,128] # the numbers of the units in the hidden layers

        # fully connected layers
        self.fc1 = nn.Linear(self.state_size, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], self.action_size)
        # batch normalization (note used currently)
        self.bn1 = nn.BatchNorm1d(hidden_units[0])
        self.bn2 = nn.BatchNorm1d(hidden_units[1])

        # the initialization of the weights
        self.reset_parameters()

    def reset_parameters(self):
        """
        This method is used for initializing the weights of
        the neural networks.

        """

        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """
        Forwarding.

        (input)
        - state: state tensor (size:33)
        (output)
        - x: action tensor (size:4)

        """

        x  = F.relu(self.fc1(state))
        # x = self.bn1(x)
        x = F.relu(self.fc2(x))
        # x = self.bn2(x)
        x = F.tanh(self.fc3(x))

        return x


class Critic(nn.Module):
    """
    This module defines a neural network model for the cricic model.
    The neural network takes state and action as an input and returns
    a Q-value.

    """

    def __init__(self, state_size, action_size, seed = 10):
        """
        Initialization.

        (input)
        - state_size (int): state size (=33)
        - action_size (int): action size (=4)
        - seed (int): random seed

        """

        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed) # random seed
        self.state_size = state_size # state size
        self.action_size = action_size # action size
        hidden_units = [128, 128] # the numbers of units in the hidden layers

        # fully connected layers
        self.fc1 = nn.Linear(self.state_size, hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0] + self.action_size, hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], 1)
        # batch normalization (not used currently)
        self.bn2 = nn.BatchNorm1d(hidden_units[1])

        # initialize the weights of the layers
        self.reset_parameters()

    def reset_parameters(self):
        """
        This method is for initializing the weights of the neural network.

        """

        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """
        Forwarding.

        (input)
        - state: state tensor (size:33)
        - action: action tensor (size:4)
        (output)
        - x: Q-value tensor (size:1)

        """

        xs = F.relu(self.fc1(state))
        x = torch.cat((xs, action), dim = 1)
        x = F.relu(self.fc2(x))
        # x = self.bn2(x)
        x = self.fc3(x)

        return x
