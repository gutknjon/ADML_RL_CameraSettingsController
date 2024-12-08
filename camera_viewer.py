import cv2
from argparse import ArgumentParser
from dataclasses import dataclass
import logging
from config import Config, EnvironmentConfig

class Camera:

    @dataclass
    class Property:
        prop_id: int
        name: str
        min_val: float
        max_val: float
        def_val: float

        def normalized_to_value(self, trackbar_val):
            '''Map normalized value from [0, 1] to [min_val, max_val]'''
            return self.min_val + (self.max_val - self.min_val) * trackbar_val
        
        def value_to_normalized(self, value = None):
            '''Map value from [min_val, max_val] to [0, 1]'''
            if value is None:
                value = self.def_val
            return (value - self.min_val) / (self.max_val - self.min_val)

        def __str__(self):
            return f'{self.name}: {self.def_val} (min: {self.min_val}, max: {self.max_val})'

    def __init__(self, camera_idx=None, parameters=[]):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info('Initializing Camera')

        # select camera
        self.camera_idx = self.select_camera(camera_idx)
        self.logger.info(f'Selected camera {self.camera_idx}')
        
        # open camera
        self.cap = cv2.VideoCapture(self.camera_idx)
        if not self.cap.isOpened():
            self.logger.error(f'Failed to open camera {self.camera_idx}')
            raise Exception(f'Failed to open camera {self.camera_idx}')

        # set camera resolution to 640x480
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # get camera properties
        self.properties = self.get_camera_properties(self.cap, parameters)
        self.logger.info(f'Camera properties: \r\n{self}')

        # create sift object
        self.sift = cv2.SIFT_create()

    def __del__(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        self.logger.info('Destroying Camera')
        pass

    def __str__(self):
        return '\n'.join([str(prop) for prop in self.properties])

    @staticmethod
    def get_camera_properties(cap, parameters):
        '''Get camera properties
        
        Args:
            cap: cv2.VideoCapture object
            parameters (list): List of camera parameters
        
        Returns:
            list: List of Property objects
        '''

        def infer_property_range(cap, prop_id, min_val=-100, max_val=100, step=1):
            # Find minimum
            current_val = cap.get(prop_id)
            min_supported = max_supported = current_val

            # Decrease until it stops changing
            for val in range(int(current_val), min_val-1, -step):
                if not cap.set(prop_id, val):
                    break
                min_supported = val

            # Increase until it stops changing
            for val in range(int(current_val), max_val+1, step):
                if not cap.set(prop_id, val):
                    break
                max_supported = val

            # Reset to initial value
            cap.set(prop_id, current_val)

            return min_supported, max_supported, current_val

        # get properties
        properties = []
        for prop_name in parameters:
            prop_id = getattr(cv2, prop_name, None)
            if prop_id is not None:
                value = cap.get(prop_id)
                if value is not None and value != -1:
                    min_val, max_val, def_val = infer_property_range(cap, prop_id, -255, 255)
                    properties.append(Camera.Property(prop_id, prop_name, min_val, max_val, def_val))

        return properties

    def get_settings(self):
        '''Get camera settings
        
        Returns:
            values (list): List of camera settings values normalized to [0, 1]
            names (list): List of camera settings names
        '''
        names = [prop.name for prop in self.properties]
        values = [prop.value_to_normalized(self.cap.get(prop.prop_id)) for prop in self.properties]
        return values, names

    def set_settings(self, values):
        '''Set camera settings
        
        Args:
            values (list): List of camera settings values normalized to [0, 1]
        '''
        for prop, value in zip(self.properties, values):
            self.cap.set(prop.prop_id, prop.normalized_to_value(value))
        return

    def get_valid_actions(self):
        '''Get valid actions
        
        Returns:
            list: List of valid actions
        '''
        return [(0, 1)]*len(self.properties)

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.logger.error('Failed to get frame')
            return None
        kp, _ = self.sift.detectAndCompute(frame, None)
        return frame, kp

    def select_camera(self, camera_idx=None):
        ''' Select camera

        Checks if a given camera index is available, 
        otherwise prompts the user to select one out of available cameras.

        Args:
            camera_idx (int): Camera index
        
        Returns:
            int: Selected camera index
        '''

        def scan_cameras(idx_range=10):
            # get available cameras
            cameras = []
            for i in range(idx_range):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    cameras.append(i)
                    cap.release()
            return cameras

        # select camera index
        if camera_idx is None:
            cameras = scan_cameras()
            if len(cameras) == 0:
                self.logger.error('No camera found')
                raise Exception('No camera found')
            
            # let user select camera
            while True:
                print('Available cameras:', cameras)
                camera_idx = int(input('Select camera: '))
                if camera_idx in cameras:
                    break
                print('Invalid camera index')
        else:
            cap = cv2.VideoCapture(camera_idx)
            if not cap.isOpened():
                self.logger.error(f'Failed to open camera {camera_idx}')
                raise Exception(f'Failed to open camera {camera_idx}')
        return camera_idx

class UserInterface:
    window_name = 'Camera Viewer'

    def __init__(self, settings, names, render=True):
        '''Initialize UserInterface
        
        Creates a window with trackbars for each property
        
        Args:
            settings (list): List of user settings values normalized to [0, 1]
            names (list): List of user settings names
        '''
        # setup logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info('Initializing UserInterface')

        # create window
        self.render = render
        self.properties = names
        if self.render:
            self.create_window(settings, names)

        pass

    def __del__(self):
        cv2.destroyAllWindows()
        self.logger.info('Destroying UserInterface')
        pass
    
    def create_window(self, settings, names):
        '''Create a window'''

        cv2.namedWindow(self.window_name)
        def nothing(x):
            pass
        for n, v in zip(names, settings):
            assert 0 <= v <= 1, f'Invalid value {v} for property {n}'
            cv2.createTrackbar(n, self.window_name, int(round(v*255)) , 255, nothing)
        return
    
    def show_frame(self, frame, features):
        '''Show frame with features
        
        Args:
            frame (np.array): Frame to show
            features (list): List of features to draw
        
        Returns:
            None
        '''
        if not self.render:
            return
        if frame is None:
            return
        frame = cv2.drawKeypoints(frame, features, None)
        cv2.putText(frame, f'Number of features: {len(features)}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
        cv2.imshow(self.window_name, frame)
        return

    def get_settings(self):
        '''Get user settings

        Args:
            None
        Returns:
            values (list): List of user settings values normalized to [0, 1]
            names (list): List of user settings names
        '''
        if not self.render:
            return [], []
        
        values = [cv2.getTrackbarPos(prop, self.window_name)/255 for prop in self.properties]
        return values, self.properties
    
    def set_settings(self, settings):
        '''Set user settings

        Args:
            settings (list): List of user settings values normalized to [0, 1]
        Returns:
            None
        '''
        if not self.render:
            return
        for prop, value in zip(self.properties, settings):
            cv2.setTrackbarPos(prop, self.window_name, int(round(value*255)))
        return

class CameraViewer:

    def __init__(self, config: EnvironmentConfig):
        # initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info('Initializing CameraViewer')

        # create camera object
        self.cam = Camera(config.camera, config.parameters)

        # create user interface
        settings, names = self.cam.get_settings()
        self.ui = UserInterface(settings, names, config.render)

        pass

    def __del__(self):
        # check if cam exists and is not none
        if hasattr(self, 'cam') and self.cam is not None:
            del self.cam
        # check if ui exists and is not none
        if hasattr(self, 'ui') and self.ui is not None:
            del self.ui
        self.logger.info('Destroying CameraViewer')
        pass

    def run(self, agent=None):
        '''Run camera viewer

        This function runs an infinite loop to get frames from the camera, and display them.
        The camera settings are updated by the agent, if provided, otherwise by the user interface.

        Args:
            agent (Agent): Agent object to control camera settings
        Returns:
            None
        '''
        while True:
            
            # get and plot frame with features
            frame, features = self.cam.get_frame()
            self.ui.show_frame(frame, features)

            # update camera settings
            if agent is not None:
                settings = agent.select_settings(state = frame)
                self.ui.set_settings(settings)
            elif self.ui is not None:
                settings, _ = self.ui.get_settings()
            else:
                raise Exception('User interface and agent are not set')

            self.cam.set_settings(settings)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        pass

def main(args, config: Config):
 
    try:
        viewer = CameraViewer(config.environment)
        viewer.run()

    except Exception as e:
        print(e)

    finally:
        del viewer
        print('Exit')

    pass

if __name__ == '__main__':

    # parse arguments
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', type=str, help='Config file')
    args = parser.parse_args()

    # setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # load config
    config = Config(args.config)

    # run main
    main(args, config)

    pass