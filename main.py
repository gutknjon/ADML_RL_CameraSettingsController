from agent import Agent
from camera_viewer import CameraViewer
import logging
import argparse

def main(args):

    # add main logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("Main")

    try:
        viewer = CameraViewer(args.camera)
        q_agent = Agent()

        if args.train:
            logger.info('Train agent start')
            q_agent.learn()
            logger.info('Train agent end')

        if args.run:
            logger.info('Run agent start')
            viewer.run(q_agent)
            logger.info('Run agent end')


    except Exception as e:
        logger.error(e)
        print(e)

    finally:
        del viewer, q_agent
        logger.info('Exit')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera Viewer')
    parser.add_argument('-c', '--camera', type=int, help='Camera index')
    parser.add_argument('-t', '--train', action='store_true', help='Train camera settings controller')
    parser.add_argument('-r', '--run', action='store_true', help='Run camera settings controller')
    args = parser.parse_args()

    main(args)