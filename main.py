import logging
import argparse
from tqdm import tqdm
import cv2
import os
import subprocess
from datetime import datetime
import mlflow
import numpy as np

from src.agent import QAgent, HumanAgent
from src.camera_viewer import CameraViewer, Camera
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

        # training loop
        for i_episode in tqdm(range(int(config.training.num_episodes)), desc='Training', unit='sequences', position=0):

            for i_sequence in tqdm(range(int(config.training.num_steps)), desc='Sequence', unit='steps', position=1):

                # epsilon decay
                eps = max(config.training.epsilon_end, config.training.epsilon_start * (config.training.epsilon_decay ** i_episode))

                # select action and update UI
                action = agent.select_settings(state = state, epsilon=eps)
                viewer.ui.set_settings(action)
                viewer.cam.set_settings(action)

                # delay for camera to adjust
                for _ in range(5):
                    viewer.cam.get_frame()               

                # observe the next state and update UI
                next_state, next_features = viewer.cam.get_frame()
                viewer.ui.show_frame(next_state, next_features)

                # update memory
                reward, kp_len, kp_resp = Camera.calculate_reward(next_features)
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
            mlflow.log_metric('keypoints_len', kp_len, step=i_episode)
            mlflow.log_metric('keypoints_resp', kp_resp, step=i_episode)
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

def grid_search(config:dict):
    """ Run grid search for camera parameters
    """
    path_results = 'grid_results.npy'

    if not os.path.exists(path_results):
        # create viewer
        viewer = CameraViewer(config.environment)

        # create grid
        _, parameters = viewer.cam.get_settings()
        N = 21  # number of grid points
        M = 10  # number of samples per grid point to average

        # create N^3 grid filled with nan
        kp_len = np.full((N, N, N), np.nan, dtype=int)
        kp_resp = np.full((N, N, N), np.nan, dtype=np.float32)
        p0_range = np.linspace(0, 1, N)
        p1_range = np.linspace(0, 1, N)
        p2_range = np.linspace(0, 1, N)

        for i0 in tqdm(range(N), desc=parameters[0], position=0, leave=False):
            for i1 in tqdm(range(N), desc=parameters[1], position=1, leave=False):
                for i2 in tqdm(range(N), desc=parameters[2], position=2, leave=False):
                    viewer.cam.set_settings([p0_range[i0], p1_range[i1], p2_range[i2]])
                    viewer.ui.set_settings([p0_range[i0], p1_range[i1], p2_range[i2]])

                    tmp_kp_len = np.full(M, np.nan)
                    tmp_kp_resp = np.full(M, np.nan)
                    for i in range(M):
                        frame, features = viewer.cam.get_frame()
                        _, tmp_kp_len[i], tmp_kp_resp[i] = Camera.calculate_reward(features)
                    viewer.ui.show_frame(frame, features)

                    kp_len[i0, i1, i2] = np.nanmean(tmp_kp_len)
                    kp_resp[i0, i1, i2] = np.nanmean(tmp_kp_resp)

                    # Break the loop on 'q' key press
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        # save results
        grid_results = {
            'kp_len': kp_len,
            'kp_resp': kp_resp,
            'p0_range': p0_range,
            'p1_range': p1_range,
            'p2_range': p2_range,
            'p0': parameters[0],
            'p1': parameters[1],
            'p2': parameters[2]
        }
        np.save('grid_results.npy', grid_results)

    else:
        grid_results = np.load(path_results, allow_pickle=True).item()


    # print max results
    max_idx_kp_len = np.unravel_index(np.nanargmax(grid_results['kp_len']), grid_results['kp_len'].shape)
    print(f'Max keypoints length:')
    print(f'Value: {grid_results["kp_len"][max_idx_kp_len]}')
    print(f'{grid_results["p0"]}: {grid_results["p0_range"][max_idx_kp_len[0]]}')
    print(f'{grid_results["p1"]}: {grid_results["p1_range"][max_idx_kp_len[1]]}')
    print(f'{grid_results["p2"]}: {grid_results["p2_range"][max_idx_kp_len[2]]}')

    max_idx_kp_resp = np.unravel_index(np.nanargmax(grid_results['kp_resp']), grid_results['kp_resp'].shape)
    print(f'Max keypoints response:')
    print(f'Value: {grid_results["kp_resp"][max_idx_kp_resp]}')
    print(f'{grid_results["p0"]}: {grid_results["p0_range"][max_idx_kp_resp[0]]}')
    print(f'{grid_results["p1"]}: {grid_results["p1_range"][max_idx_kp_resp[1]]}')
    print(f'{grid_results["p2"]}: {grid_results["p2_range"][max_idx_kp_resp[2]]}')

    # plot results
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 6))

    # First subplot for Keypoints length
    ax1 = fig.add_subplot(121, projection='3d')
    X, Y, Z = np.meshgrid(grid_results['p0_range'], grid_results['p1_range'], grid_results['p2_range'])
    ax1.scatter(X, Y, Z, c=grid_results['kp_len'], cmap='viridis')
    ax1.set_xlabel(grid_results['p0'])
    ax1.set_ylabel(grid_results['p1'])
    ax1.set_zlabel(grid_results['p2'])
    ax1.set_title('Keypoints length')

    # Second subplot for Keypoints response
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(X, Y, Z, c=grid_results['kp_resp'], cmap='viridis')
    ax2.set_xlabel(grid_results['p0'])
    ax2.set_ylabel(grid_results['p1'])
    ax2.set_zlabel(grid_results['p2'])
    ax2.set_title('Keypoints response')

    plt.suptitle('3D Scatter Plots of Grid Search Results')

    # create two plots which shows the mean and std value of grid_results['p2'] as a heatmap for grid_results['p0'] and grid_results['p1']
    def show_heatmap(axis, data, title):
        axis.imshow(data, cmap='viridis')
        axis.set_title(title)
        axis.set_xlabel(grid_results['p0'])
        axis.set_xticks(np.arange(len(grid_results['p0_range']))[::2])
        axis.set_xticklabels([f'{p:.2f}' for p in grid_results['p0_range'][::2]])
        axis.set_ylabel(grid_results['p1'])
        axis.set_yticks(np.arange(len(grid_results['p1_range']))[::2])
        axis.set_yticklabels([f'{p:.2f}' for p in grid_results['p1_range'][::2]])
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    show_heatmap(ax[0], grid_results['kp_len'][:,:,max_idx_kp_len[2]], 
                 f'Keypoints length ({grid_results["p2"]} = {grid_results["p2_range"][max_idx_kp_len[2]]:.02f}')
    show_heatmap(ax[1], grid_results['kp_resp'][:,:,max_idx_kp_resp[2]], 
                 f'Keypoints response ({grid_results["p2"]} = {grid_results["p2_range"][max_idx_kp_resp[2]]:.02f}')
    
    fig.suptitle('Grid search results')
    # fig.tight_layout()

    plt.show()

    pass

def main(args, config):

    # try:
    if args.train:
        train(config)

    if args.run:
        run(config)

    if args.grid_search:
        grid_search(config)

    # except Exception as e:
    #     logging.error(e)
    #     print(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera Viewer')
    parser.add_argument('-c', '--config', type=str, help='Config file')
    parser.add_argument('-t', '--train', action='store_true', help='Train camera settings controller')
    parser.add_argument('-r', '--run', action='store_true', help='Run camera settings controller')
    parser.add_argument('-g', '--grid_search', action='store_true', help='Run grid search for camera parameters')
    args = parser.parse_args()

    # setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    config = Config(args.config)
    main(args, config)