from agent import QAgent, HumanAgent
from camera_viewer import CameraViewer
from time import time
import logging
import argparse
from tqdm import tqdm
import cv2

BATCH_SIZE = 150
BATCHES_TO_QTARGET_SWITCH = 1000
GAMMA = 0.95
TAU = 1
MEMORY_SIZE = 100000
LEARNING_RATE = 0.0001

def train(viewer: CameraViewer, num_of_sequences=1e6, savedir='./models'):
    """ Train the agent 
    """
    logging.info('Train agent start')


    agent = QAgent(viewer, memory_size=MEMORY_SIZE)

    train_start_time = time()
    for i in tqdm(range(int(num_of_sequences)), desc='Training', unit='batch'):

        state, reward = viewer.cam.get_frame()
        viewer.ui.show_frame(state, reward)

        action = agent.select_settings(state)
        viewer.cam.set_settings(action)

        next_state, reward = viewer.cam.get_frame()
        viewer.ui.show_frame(state, reward)

        agent.add_to_memory(state, action, next_state, len(reward))
        agent.learn(learning_rate = LEARNING_RATE)

        if i % BATCHES_TO_QTARGET_SWITCH == 0:
            # agent.switch_q_target()
            pass

        if i % 1000 == 0 and i > 0:
            agent.save(savedir)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def run(viewer: CameraViewer, model_path):
    """ Run the agent 
    """
    if model_path is None:
        agent = HumanAgent(viewer)
    else:
        agent = QAgent(viewer, memory_size=MEMORY_SIZE)
        # agent.restore(model_path)

    # run the agent
    viewer.run(agent)

    pass

def main(args):

    # add main logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # try:
    viewer = CameraViewer(args.camera)

    if args.train:
        train(viewer, num_of_sequences=100)

    if args.run:
        run(viewer, args.model)


    # except Exception as e:
    #     logging.error(e)
    #     print(e)

    # finally:
    del viewer
    logging.info('Exit')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera Viewer')
    parser.add_argument('-c', '--camera', type=int, help='Camera index')
    parser.add_argument('-t', '--train', action='store_true', help='Train camera settings controller')
    parser.add_argument('-r', '--run', action='store_true', help='Run camera settings controller')
    parser.add_argument('-m', '--model', type=str, help='Model path')
    args = parser.parse_args()

    main(args)