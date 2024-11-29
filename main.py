from agent import QAgent, HumanAgent
from camera_viewer import CameraViewer
from time import time
import logging
import argparse
from tqdm import tqdm
import numpy as np
import cv2
import os
import git
from datetime import datetime

import matplotlib.pyplot as plt

from config import load_config, save_config


class LearningVisualizer:
    def __init__(self, action_names):
        """
        Initializes the LearningVisualizer class.

        Parameters:
        action_names (list): List of action names to track.
        """
        self.action_names = action_names
        self.num_actions = len(action_names)

        # Data storage
        self.action_data = [[] for _ in range(self.num_actions)]
        self.reward_data = []
        self.epsilon_data = []

        # Create the figure and axes
        self.fig, (self.ax_actions, self.ax_rewards, self.ax_epsilon) = plt.subplots(3, 1, figsize=(10, 8))

        # Setup action plot
        self.action_lines = [self.ax_actions.plot([], [], label=name)[0] for name in action_names]
        self.ax_actions.set_title("Action Parameters")
        self.ax_actions.set_ylabel("Values")
        self.ax_actions.legend()

        # Setup reward plot
        self.reward_line, = self.ax_rewards.plot([], [], label="Reward", color="orange")
        self.ax_rewards.set_title("Reward")
        self.ax_rewards.set_ylabel("Reward")

        # Setup epsilon plot
        self.epsilon_line, = self.ax_epsilon.plot([], [], label="Epsilon", color="green")
        self.ax_epsilon.set_title("Epsilon")
        self.ax_epsilon.set_xlabel("Steps")
        self.ax_epsilon.set_ylabel("Epsilon")

        # Adjust layout
        plt.tight_layout()

    def update(self, actions, reward, epsilon):
        """
        Updates the plots with new action parameters, reward, and epsilon.

        Parameters:
        actions (np.array): Array of action parameter values.
        reward (float): Reward value.
        epsilon (float): Epsilon value.
        """
        # Update data
        for i in range(self.num_actions):
            self.action_data[i].append(actions[i])
        self.reward_data.append(reward)
        self.epsilon_data.append(epsilon)

        # Update action plots
        for i, line in enumerate(self.action_lines):
            line.set_xdata(range(len(self.action_data[i])))
            line.set_ydata(self.action_data[i])
            self.ax_actions.relim()
            self.ax_actions.autoscale_view()

        # Update reward plot
        self.reward_line.set_xdata(range(len(self.reward_data)))
        self.reward_line.set_ydata(self.reward_data)
        self.ax_rewards.relim()
        self.ax_rewards.autoscale_view()

        # Update epsilon plot
        self.epsilon_line.set_xdata(range(len(self.epsilon_data)))
        self.epsilon_line.set_ydata(self.epsilon_data)
        self.ax_epsilon.relim()
        self.ax_epsilon.autoscale_view()

        # Redraw the figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # Pause to allow for smooth plotting
        plt.pause(0.001)

    def save(self, path):
        """
        Saves the current plot to a file.

        Parameters:
        path (str): Path to save the plot to.
        """
        self.fig.savefig(path)

def train(viewer: CameraViewer, config:dict):
    """ Train the agent 
    """
    logging.info(f'Training start using {config["training"]["num_episodes"]} episodes and {config["training"]["num_steps"]} steps per episode')

    agent = QAgent(viewer, config['agent'])

    # create learning visualizer
    _, parameters = viewer.cam.get_settings()
    visualizer = LearningVisualizer(parameters)

    state, features = viewer.cam.get_frame()
    viewer.ui.show_frame(state, features)

    for i_episode in tqdm(range(int(config['training']['num_episodes'])), desc='Training', unit='sequences', position=0):

        for i_sequence in tqdm(range(int(config['training']['num_steps'])), desc='Sequence', unit='steps', position=1):

            # epsilon decay
            # eps =   config['training']['epsilon_end'] + \
            #         (config['training']['epsilon_start'] - config['training']['epsilon_end']) * \
            #         np.exp(-1. * i_steps / config['training']['epsilon_decay'])
            
            eps = max(config['training']['epsilon_end'], config['training']['epsilon_start'] * (config['training']['epsilon_decay'] ** i_episode))

            # select action and update UI
            action = agent.select_settings(state, epsilon=eps)
            viewer.cam.set_settings(action)
            viewer.ui.set_settings(action)

            # observe the next state and update UI
            for i in range(10):
                next_state, next_features = viewer.cam.get_frame()
            viewer.ui.show_frame(next_state, next_features)

            # update memory
            reward = len(next_features) - len(features)
            agent.add_to_memory(state, action, next_state, reward)

            # optimize the model
            agent.learn(batch_size = config["training"]["batch_size"])

            # update state
            state = next_state
            features = next_features

            # draw the frame
            cv2.waitKey(1)

        # update visualizer
        visualizer.update(action, len(features), eps)
        logging.info(f'Episode {i_episode} completed')

        if i_episode % config['training']['chkpt_interval'] == 0 and i_episode > 0 and config['_log_dir'] is not None:
            name = os.path.join(config['_log_dir'], 'checkpoints', f"{os.path.basename(config['_log_dir'])}_chkpt_{i_episode}.pth")
            logging.info(f'Saving checkpoint to {name}')
            agent.save(name)
            visualizer.save(name.replace('.pth', '.png'))

def run(viewer: CameraViewer, agent_config:dict):
    """ Run the agent 
    """

    if agent_config['type'] == 'HumanAgent':
        agent = HumanAgent(viewer)
    elif agent_config['type'] == 'QAgent':
        agent = QAgent(viewer, config['agent'])
        
        if agent_config['checkpoint'] is not None:
            agent.restore(agent_config['checkpoint'])
    else:
        raise ValueError(f'Unknown agent type {agent_config["type"]}')

    # run the agent
    viewer.run(agent)

    pass

def main(args, config):

    # create log name containing of datetime followed by the experiment name
    if config['logging']['log_dir'] is None:
        log_dir = None
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        log_dir = os.path.join(config['logging']['log_dir'], f"{datetime.now().strftime('%y%m%d_%H%M%S')}_{config['experiment_name']}")
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(filename=os.path.join(log_dir, "log.txt"), level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # serialize config to log folder
        save_config(config, os.path.join(log_dir, f"config.yaml"))
    config["_log_dir"] = log_dir

    # log git commit short hash
    repo = git.Repo(search_parent_directories=True)
    commit = repo.head.object.hexsha
    logging.info(f'Training repository commit: {commit}')

    # try:
    viewer = CameraViewer(config['environment'])

    if args.train:
        train(viewer, config)

    if args.run:
        run(viewer, config['agent'])


    # except Exception as e:
    #     logging.error(e)
    #     print(e)

    # finally:
    del viewer
    logging.info('Exit')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera Viewer')
    parser.add_argument('-c', '--config', type=str, help='Config file')
    parser.add_argument('-t', '--train', action='store_true', help='Train camera settings controller')
    parser.add_argument('-r', '--run', action='store_true', help='Run camera settings controller')
    args = parser.parse_args()

    config = load_config(args.config)
    main(args, config)