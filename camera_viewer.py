import cv2
from argparse import ArgumentParser
from dataclasses import dataclass
import logging

class CameraProperties:

    @dataclass
    class Property:
        prop_id: int
        name: str
        min_val: float
        max_val: float
        def_val: float

        def trackbar_to_value(self, trackbar_val):
            # map trackbar value from [0, 255] to [min_val, max_val]
            return self.min_val + (self.max_val - self.min_val) * trackbar_val / 255
        
        def value_to_trackbar(self, value = None):
            # map value from [min_val, max_val] to [0, 255]
            if value is None:
                value = self.def_val
            return int(255 * (value - self.min_val) / (self.max_val - self.min_val))

        def __str__(self):
            return f'{self.name}: {self.def_val} (min: {self.min_val}, max: {self.max_val})'

    def __init__(self, cap):
        self.cap = cap
        property = self.get_camera_properties(cap)
        self.properties = []
        for prop_name, prop_id in property.items():
            min_val, max_val, def_val = self.infer_property_range(cap, prop_id, -255, 255)
            self.properties.append(self.Property(prop_id, prop_name, min_val, max_val, def_val))

    def __str__(self):
        return '\n'.join([str(prop) for prop in self.properties])

    @staticmethod
    def get_camera_properties(cap):
        camera_properties = {
            "CAP_PROP_BRIGHTNESS": cv2.CAP_PROP_BRIGHTNESS,
            "CAP_PROP_CONTRAST": cv2.CAP_PROP_CONTRAST,
            # "CAP_PROP_SATURATION": cv2.CAP_PROP_SATURATION,
            # "CAP_PROP_GAIN": cv2.CAP_PROP_GAIN,
            "CAP_PROP_EXPOSURE": cv2.CAP_PROP_EXPOSURE,
            # "CAP_PROP_AUTOFOCUS": cv2.CAP_PROP_AUTOFOCUS,
            # "CAP_PROP_AUTO_EXPOSURE": cv2.CAP_PROP_AUTO_EXPOSURE,
        }

        # get properties
        properties = {}
        for prop_name, prop_id in camera_properties.items():
            prop_id = getattr(cv2, prop_name, None)
            if prop_id is not None:
                value = cap.get(prop_id)
                if value is not None and value != -1:
                    properties[prop_name] = prop_id

        return properties

    @staticmethod
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

class CameraViewer:

    def __init__(self, camera_idx=None):
        # initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info('Initializing CameraViewer')

        # select camera
        self.camera_idx = self.select_camera(camera_idx)
        self.logger.info(f'Selected camera {self.camera_idx}')
        
        # open camera
        self.cap = cv2.VideoCapture(self.camera_idx)
        if not self.cap.isOpened():
            self.logger.error(f'Failed to open camera {self.camera_idx}')
            raise Exception(f'Failed to open camera {self.camera_idx}')
        
        # get camera properties
        self.properties = CameraProperties(self.cap)
        self.logger.info(f'Camera properties: \r\n{self.properties}')

        # create sift object
        self.sift = cv2.SIFT_create()

        pass

    def __del__(self):
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.logger.info('Destroying CameraViewer')
        pass

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.logger.error('Failed to get frame')
            return None
        kp, _ = self.sift.detectAndCompute(frame, None)
        return frame, kp
    
    def set_settings(self, settings):
        for prop in self.properties.properties:
            value = settings.get(prop.name, None)
            if value is not None:
                self.cap.set(prop.prop_id, value)
                cv2.setTrackbarPos(prop.name, 'Camera Viewer', prop.value_to_trackbar(value))
        return
    
    def get_settings(self):
        settings = {}
        for prop in self.properties.properties:
            value = prop.trackbar_to_value(cv2.getTrackbarPos(prop.name, 'Camera Viewer'))
            settings[prop.name] = value
        return settings
    
    def show_frame(self, frame, features):
        if frame is None:
            return
        frame = cv2.drawKeypoints(frame, features, None)
        cv2.putText(frame, f'Number of features: {len(features)}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)
        cv2.imshow('Camera Viewer', frame)
        return
    
    def create_window(self):
        # Create a window
        cv2.namedWindow('Camera Viewer')
        def nothing(x):
            pass
        for prop in self.properties.properties:
            cv2.createTrackbar(prop.name, 'Camera Viewer', prop.value_to_trackbar() , 255, nothing)
        return
    
    def run(self, agent=None):

        # Create a window
        self.create_window()

        while True:
            
            # get and plot frame with features
            frame, features = self.get_frame()
            self.show_frame(frame, features)

            # update camera settings
            if agent is not None:
                # TODO: Implement agent
                settings = self.get_settings()
                pass
            else:
                settings = self.get_settings()
            self.set_settings(settings)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        pass

    def select_camera(self, camera_idx=None):
        # select camera index
        if camera_idx is None:
            cameras = self.scan_cameras()
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

    @staticmethod
    def scan_cameras(idx_range=10):
        # get available cameras
        cameras = []
        for i in range(idx_range):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cameras.append(i)
                cap.release()
        return cameras

def main(args):
    try:
        viewer = CameraViewer(args.camera)
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
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    args = parser.parse_args()

    # run main
    main(args)

    pass