import random
from dqn import DeepQNetworkModel
from camera_viewer import CameraViewer
from memory_buffers import ReplayMemory
import cv2
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
    def __init__(self, camera_viewer: CameraViewer, memory_size=1000, **kwargs):
        super().__init__(camera_viewer)
        self.name = self.__class__.__name__

        self.valid_actions = self.cam_viewer.cam.get_valid_actions()
        frame, _ = self.cam_viewer.cam.get_frame()

        self.eps = 0.5
        self.dqn = DeepQNetworkModel(frame.shape, len(self.valid_actions), ReplayMemory(memory_size))

    def select_settings(self, frame, **kwargs):
        return self.dqn.act(frame, self.eps)

    def learn(self, **kwargs):
        return self.dqn.learn(learning_rate=kwargs['learning_rate'])

    def add_to_memory(self, state, action, next_state, reward):
        self.dqn.memory.push(state, action, next_state, reward)

    def save(self, filename):
        self.dqn.save(filename)
        return 

    def restore(self, filename):
        self.dqn.restore(filename)
        return
