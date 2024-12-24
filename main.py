import logging
import argparse
from tqdm import tqdm
import cv2
import os
import subprocess
from datetime import datetime
import mlflow

from src.agent import QAgent, HumanAgent
from src.camera_viewer import CameraViewer
from src.config import Config

def get_short_sha(repo_path="."):
    """
    Get the short SHA of the latest commit in a Git repository.

    Args:
        repo_path (str): Path to the Git repository (default is the current directory).
    
    Returns:
        str: The short SHA of the latest commit.
    """
    try:
        # Run the `git rev-parse --short HEAD` command
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_path,  # Specify the repository path
            text=True       # Decode the output to a string
        ).strip()
        return sha
    except subprocess.CalledProcessError as e:
        # Handle errors (e.g., not a git repository)
        print(f"Error: {e}")
        return None

def create_agent(config:dict, viewer:CameraViewer):
    # create agent
    if config.agent.type == 'HumanAgent':
        agent = HumanAgent(viewer)
    elif config.agent.type == 'QAgent':
        agent = QAgent(viewer, config = config.agent)
        
        if config.agent.checkpoint is not None:
            agent.restore(config.agent.checkpoint)
    else:
        raise ValueError(f'Unknown agent type {config.agent.type}')
    
    return agent


def train(config:Config):
    """ Train the agent 
    """

    # setup logging
    run_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    if config.logging.run_name is not None:
        run_name = run_name + '_' + config.logging.run_name
    
    log_dir = None
    log_file = None
    if config.logging.log_dir is not None:
        log_dir = os.path.join(config.logging.log_dir, run_name)
        log_file = os.path.join(log_dir, 'training.log')
        os.makedirs(log_dir, exist_ok=True)
    
    # set logging to file
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename=log_file, force=True)
    logging.info(f'Training start using {config.training.num_episodes} episodes and {config.training.num_steps} steps per episode')

    # create viewer
    viewer = CameraViewer(config.environment)

    # create agent
    agent = create_agent(config, viewer)

    # create learning visualizer
    _, cam_parameters = viewer.cam.get_settings()
    state, features = viewer.cam.get_frame()
    viewer.ui.show_frame(state, features)

    # set up tracking
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    mlflow.set_experiment(config.logging.experiment_name)
    with mlflow.start_run(run_name=run_name):

        # log parameters
        mlflow.log_params(config.__dict__)
        mlflow.set_tag(*config.logging.tags)

        # log artifacts
        # TODO: log artifacts

        for i_episode in tqdm(range(int(config.training.num_episodes)), desc='Training', unit='sequences', position=0):

            for i_sequence in tqdm(range(int(config.training.num_steps)), desc='Sequence', unit='steps', position=1):

                # epsilon decay
                eps = max(config.training.epsilon_end, config.training.epsilon_start * (config.training.epsilon_decay ** i_episode))

                # select action and update UI
                action = agent.select_settings(state = state, epsilon=eps)
                viewer.cam.set_settings(action)
                viewer.ui.set_settings(action)

                # observe the next state and update UI
                next_state, next_features = viewer.cam.get_frame()
                viewer.ui.show_frame(next_state, next_features)

                # update memory
                reward = len(next_features)
                agent.add_to_memory(state = state, action = action, next_state = next_state, reward = reward)

                # optimize the model
                agent.learn(batch_size = config.training.batch_size)

                # update state
                state = next_state
                features = next_features

                # draw the frame
                cv2.waitKey(1)

            logging.info(f'Episode {i_episode} completed')

            # log metrics
            mlflow.log_metric('reward', reward, step=i_episode)
            mlflow.log_metric('epsilon', eps, step=i_episode)
            for i, p in enumerate(cam_parameters):
                mlflow.log_metric(p, action[i], step=i_episode)

        # save final checkpoint
        name = f"{os.path.basename(log_dir)}_final.pth"
        agent.save(os.path.join(log_dir, name))
        mlflow.log_artifact(local_path=os.path.join(log_dir, name))
        mlflow.log_artifact(local_path=config.path)

        # flush logging
        logging.info('Training completed')
        logging.getLogger().handlers[0].flush()
        if log_file is not None:
            mlflow.log_artifact(local_path=log_file)

def run(config:dict):
    """ Run the agent 
    """

    # create viewer
    viewer = CameraViewer(config.environment)

    # create agent
    agent = create_agent(config, viewer)

    # run the agent
    viewer.run(agent)

    pass

def main(args, config):

    # try:
    if args.train:
        train(config)

    if args.run:
        run(config)

    # except Exception as e:
    #     logging.error(e)
    #     print(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera Viewer')
    parser.add_argument('-c', '--config', type=str, help='Config file')
    parser.add_argument('-t', '--train', action='store_true', help='Train camera settings controller')
    parser.add_argument('-r', '--run', action='store_true', help='Run camera settings controller')
    args = parser.parse_args()

    # setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    config = Config(args.config)
    main(args, config)