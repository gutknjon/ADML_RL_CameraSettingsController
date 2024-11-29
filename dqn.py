import random
import logging
import numpy as np
import torch

class DeepQNetworkModel:

    default_epsilon = 0.1
    epsilon = None
    min_samples_for_predictions = 5

    def __init__(self,
                 input_size,
                 output_size,
                 memory):
        """
        Create a new Deep Q Network model
        :param memory: an instance of type memory_buffers.Memory
        """
        self.input_size = input_size
        self.output_size = output_size
        self.memory = memory

        # Create the Q-Network
        self.policy_net = self.__create_q_network(input_size=input_size, output_size=output_size)
        self.target_net = self.__create_q_network(input_size=input_size, output_size=output_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        

    def __create_q_network(self, input_size, output_size):
        return QNetwork(input_size=input_size, output_size=output_size)

    def learn(self, learning_rate=None, batch_size=None):
        """
        Initialize a learning attempt
        :param learning_rate: a learning rate overriding default_learning_rate
        :param batch_size: a batch_size overriding default_batch_size
        :return: None if no learning was made, or the cost of learning if it did happen
        """
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
            x = torch.tensor(state).float().unsqueeze(0).permute(0, 3, 1, 2)
            action = self.policy_net.model(x)
            action = action[0].detach().numpy()
            logging.debug("Predicted action is %s [Epsilon = %s]",
                          action, eps)

        return action

    def add_to_memory(self, state, action, reward, next_state, is_terminal_state):
        """
        Add new state-transition to memory
        :param state: a Numpy array representing a state
        :param action: an integer representing the selected action
        :param reward: a number representing the received reward
        :param next_state: a Numpy array representing the state reached after performing the action
        :param is_terminal_state: boolean. mark state as a terminal_state. next_state will have no effect.
        """
        self.memory.append({'state': state, 'action': action, 'reward': reward,
                            'next_state': next_state, 'is_terminal': is_terminal_state})

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

class QNetwork(torch.nn.Module):
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
        self.mobilenet.classifier[1] = torch.nn.Linear(1280, output_size)

        # normalize output
        self.sigmoid = torch.nn.Sigmoid()

        # combine all layers
        self.model = torch.nn.Sequential(self.mobilenet, self.sigmoid)

    def forward(self, x):
        """ Forward pass
        :param x: input tensor of shape 224x224x3
        :return: output tensor of shape output_size
        """
        return self.model(x)
