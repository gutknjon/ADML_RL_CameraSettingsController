import random
import logging
import numpy as np
import torch

from memory_buffers import Transition, ReplayMemory

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
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        state_batch = torch.cat(batch.state).float().to(self.device)
        action_batch = torch.cat(batch.action).float().to(self.device)
        reward_batch = torch.cat(batch.reward).float().to(self.device)
        next_state_batch = torch.cat(batch.next_state).float().to(self.device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch) #.gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(batch_size, device=self.device)
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch)
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = torch.nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

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
        if rnd < eps: # or len(self.memory) < self.min_samples_for_predictions:
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
        # create mobile net v2 model
        self.mobilenet = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)

        # freeze all layers
        for param in self.mobilenet.parameters():
            param.requires_grad = False

        # replace the last layer with a new layer with output_size outputs
        self.mobilenet.classifier[1] = torch.torch.nn.Linear(1280, output_size)

        # normalize output
        self.sigmoid = torch.torch.nn.Sigmoid()

        # combine all layers
        self.model = torch.torch.nn.Sequential(self.mobilenet, self.sigmoid)

    def forward(self, x):
        """ Forward pass
        :param x: input tensor of shape 224x224x3
        :return: output tensor of shape output_size
        """
        return self.model(x)
