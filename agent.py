import random
from dqn import DeepQNetworkModel
from camera_viewer import CameraViewer
from memory_buffers import ReplayMemory
import os
import torch
import logging
from abc import abstractmethod

class BaseAgent:
    """
    Base class for all player types
    """
    name = None
    player_id = None

    def __init__(self, camera_viewer: CameraViewer):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info('Initializing Agent')

        self.cam_viewer = camera_viewer
        pass

    def shutdown(self):
        pass

    def add_to_memory(self, **kwargs):
        pass

    def save(self, filename):
        pass

    @abstractmethod
    def select_settings(self, **kwargs):
        pass

    @abstractmethod
    def learn(self, **kwargs):
        pass


class HumanAgent(BaseAgent):
    """ Human Agent """
    def __init__(self, camera_viewer: CameraViewer):
        super().__init__(camera_viewer)
        self.name = self.__class__.__name__

    def select_settings(self, **kwargs):
        settings, _ = self.cam_viewer.ui.get_settings()
        return settings

    def learn(self, **kwargs):
        pass


class QAgent(BaseAgent):
    """ QAgent """
    def __init__(self, camera_viewer: CameraViewer, **kwargs):
        super().__init__(camera_viewer)
        self.name = self.__class__.__name__

        self.valid_actions = self.cam_viewer.cam.get_valid_actions()
        frame, _ = self.cam_viewer.cam.get_frame()


        self.dqn = DeepQNetworkModel(input_size = frame.shape, 
                                     output_size = len(self.valid_actions), 
                                     learning_rate=kwargs['agent_config']['learning_rate'],
                                     gamma=kwargs['agent_config']['gamma'],
                                     memory = ReplayMemory(kwargs['agent_config']['memory_size']))

    def select_settings(self, **kwargs):
        if "epsilon" in kwargs:
            eps = kwargs['epsilon']
        else:
            eps = 0.0
        return self.dqn.act(state = kwargs['state'], epsilon = eps)

    def learn(self, **kwargs):
        return self.dqn.learn(batch_size=kwargs['batch_size'])

    def add_to_memory(self, **kwargs):
        self.dqn.add_to_memory( state = kwargs['state'], 
                                action = kwargs['action'],
                                next_state = kwargs['next_state'],
                                reward = kwargs['reward'])

    def save(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.dqn.save(filename)
        return 

    def restore(self, filename):
        self.dqn.restore(filename)
        return
