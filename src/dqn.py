import random
import logging
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import MobileNet_V3_Small_Weights
from torch import nn

from src.memory_buffers import Transition, ReplayMemory

class DeepQNetworkModel:

    default_epsilon = 0.1
    epsilon = None
    min_samples_for_predictions = 5
    device = 'cpu'

    def __init__(self,
                 input_size,
                 output_size,
                 learning_rate = 0.001,
                 gamma = 0.99,
                 tau = 0.001,
                 memory = ReplayMemory(1000)):
        """
        Create a new Deep Q Network model
        :param memory: an instance of type memory_buffers.Memory
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info('Initializing DeepQNetworkModel')

        self.input_size = input_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.memory = memory
        self.gamma = gamma
        self.tau = tau

        if torch.cuda.is_available():
            self.device = 'cuda'
        self.logger.info('Using device: %s', self.device)

        # Create the Q-Network
        self.policy_net = self.__create_q_network(input_size=input_size, output_size=output_size).to(self.device)
        self.target_net = self.__create_q_network(input_size=input_size, output_size=output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Create the optimizer
        self.optimizer = torch.optim.AdamW(self.policy_net.parameters(), lr=self.learning_rate, amsgrad=True)

    def __create_q_network(self, input_size, output_size):
        return QNetwork(input_size=input_size, output_size=output_size)

    def learn(self, batch_size=None):
        """
        Initialize a learning attempt
        :param learning_rate: a learning rate overriding default_learning_rate
        :param batch_size: a batch_size overriding default_batch_size
        :return: None if no learning was made, or the cost of learning if it did happen
        """

        if len(self.memory) < batch_size:
            return

        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        state_batch = torch.cat(batch.state).float().to(self.device)
        action_batch = torch.cat(batch.action).float().to(self.device)
        reward_batch = torch.cat(batch.reward).float().to(self.device)
        next_state_batch = torch.cat(batch.next_state).float().to(self.device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # actions taken.
        state_action_values = self.policy_net(state_batch)

        # Compute V(s_{t+1}) for all next states using target_net
        next_state_values = torch.zeros(batch_size, device=self.device)
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch)

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Calculate the loss
        criterion = torch.nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # Soft update of the target network's weights
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)

        return None

    def act(self, state, epsilon=None):
        """
        Select an action for the given state
        :param state: a Numpy array representing a state
        :param epsilon: an epsilon value to be used for the eps-greedy policy, overriding default_epsilon
        :return: a number representing the selected action
        """
        eps = epsilon if epsilon is not None else self.default_epsilon
        rnd = random.random()
        if rnd < eps: 
            # create output_size random float values in the rang [0, 1]
            action = [random.random() for _ in range(self.output_size)]
            logging.debug("Choosing a random action: %s [Epsilon = %s]", action, eps)
        else:
            self.policy_net.model.eval()
            x = torch.tensor(state).float().unsqueeze(0).permute(0, 3, 1, 2).to(self.device)
            action = self.policy_net.model(x)
            action = action[0].cpu().detach().numpy()
            logging.debug("Predicted action is %s [Epsilon = %s]",
                          action, eps)

        return action

    def add_to_memory(self, state, action, next_state, reward):
        """
        Add new state-transition to memory
        :param state: a Numpy array representing a state
        :param action: an integer representing the selected action
        :param next_state: a Numpy array representing the state reached after performing the action
        :param reward: a number representing the received reward
        """
        self.memory.push(   torch.tensor(state).permute(2,0,1).unsqueeze(0), torch.tensor(action).unsqueeze(0), 
                            torch.tensor(next_state).permute(2,0,1).unsqueeze(0), torch.tensor(reward).reshape(1,1))

    def __fetch_from_batch(self, batch, key, enum=False):
        if enum:
            return np.array(list(enumerate(map(lambda x: x[key], batch))))
        else:
            return np.array(list(map(lambda x: x[key], batch)))

    def save(self, filename):
        torch.save(self.policy_net.model.state_dict(), filename)
        return
    
    def restore(self, filename):
        self.policy_net.model.load_state_dict(torch.load(filename))
        self.target_net.model.load_state_dict(torch.load(filename))
        return

class QNetwork(torch.torch.nn.Module):
    """
    A Q-Network implementation
    """
    def __init__(self, input_size, output_size):
        super().__init__()
            
        # Load the MobileNetV3 Small model with pretrained weights
        self.mobilenet = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        # Freeze all layers (no training on pretrained weights)
        for param in self.mobilenet.parameters():
            param.requires_grad = False

        # Replace the last classifier layer with a new one for your custom output size
        self.mobilenet.classifier[3] = nn.Linear(self.mobilenet.classifier[3].in_features, output_size)

        # Normalize the output using Sigmoid activation (for binary classification, adjust if necessary)
        self.sigmoid = nn.Sigmoid()

        # Combine the layers into a final model
        self.model = nn.Sequential(
            self.mobilenet, 
            self.sigmoid)

    def forward(self, x):
        """ Forward pass
        :param x: input tensor of shape 224x224x3
        :return: output tensor of shape output_size
        """
        return self.model(x)
