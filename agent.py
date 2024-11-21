import random
from q import Q
from camera_viewer import CameraViewer
import cv2
import logging

class Agent:
    def __init__(self, camera_viewer: CameraViewer):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info('Initializing Agent')

        self.eps = 1.0
        self.qlearner = Q()

        self.cam_viewer = camera_viewer
        self.valid_actions = self.cam_viewer.cam.get_valid_actions()

    def _get_action(self, state):
        return [random.randint(min_val, max_val) for min_val, max_val in self.valid_actions]
        if random.random() < self.eps:
            # select uniform random value within valid_actions of type [(min, max), ...]
            return [random.randint(min_val, max_val) for min_val, max_val, _ in self.valid_actions]
        best = self.qlearner.get_best_action(state)
        if best is None:
            return [random.randint(min_val, max_val) for min_val, max_val, _ in self.valid_actions]
        return best

    def _learn_one_sequence(self, m=100):

        self.logger.debug(f'Learning one sequence with {m} frames')
        state, features = self.cam_viewer.cam.get_frame()
        self.cam_viewer.ui.show_frame(state, features)
        for _ in range(m):
            action = self._get_action(state)
            self.cam_viewer.cam.set_settings(action)
            self.cam_viewer.ui.set_settings(action)

            # if action != self.cam_viewer.cam.get_settings()[0]:
            #     self.logger.error(f'Action not set correctly: {action} != {self.cam_viewer.cam.get_settings()[0]}')
            state, features = self.cam_viewer.cam.get_frame()
            self.cam_viewer.ui.show_frame(state, features)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # self.qlearner.update(state, action, state, features)

    def learn(self, n=200000):
        self.logger.info(f'Learning for {n} sequences')
        for _ in range(n):
            self._learn_one_sequence(10)
            self.eps -= 0.0001
        self.logger.info('Learning end')
