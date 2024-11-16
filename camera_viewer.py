import cv2
from argparse import ArgumentParser
from dataclasses import dataclass

def select_camera():
    # get available cameras
    cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cameras.append(i)
            cap.release()

    if len(cameras) == 0:
        print('No camera found')
        return None
    else:
        print('Available cameras:', cameras)
        camera = int(input('Select camera: '))
        return camera

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
            return 
        
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
            print(f'{prop_name}: {def_val} (min: {min_val}, max: {max_val})')
            self.properties.append(self.Property(prop_id, prop_name, min_val, max_val, def_val))

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

def main(args):

    try:
        # select camera
        cam_idx = args.camera
        if args.camera is None:
            cam_idx = select_camera()
            if cam_idx is None:
                return

        # open camera
        def nothing(x):
            pass

        # Initiate SIFT detector
        sift = cv2.SIFT_create()

        # Open the camera
        # get available cameras
        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            raise Exception(f'Failed to open camera {cam_idx}')

        # Create a window
        cv2.namedWindow('Camera Viewer')

        # disable autoexposure
        # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, -255)

        # get camera properties
        properties = CameraProperties(cap)

        # add trackbars
        for prop in properties.properties:
            cv2.createTrackbar(prop.name, 'Camera Viewer', prop.value_to_trackbar() , 255, nothing)

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                break

            # update properties
            for prop in properties.properties:
                value = prop.trackbar_to_value(cv2.getTrackbarPos(prop.name, 'Camera Viewer'))
                if not cap.set(prop.prop_id, value):
                    print(f'Failed to set {prop.name} to {value}')

            # calculate sift features
            kp, _ = sift.detectAndCompute(frame, None)
            frame = cv2.drawKeypoints(frame, kp, None)

            # get the quality of each feature
            N = len(kp)

            # Display the number of features
            cv2.putText(frame, f'Number of features: {N}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_4)

            # Display the resulting frame
            cv2.imshow('Camera Viewer', frame)

            # Break the loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(e)

    finally:
        # When everything is done, release the capture
        cap.release()
        cv2.destroyAllWindows()
    
    return

if __name__ == '__main__':

    # parse arguments
    parser = ArgumentParser()
    parser.add_argument('--camera', type=int, default=0, help='Camera index')
    args = parser.parse_args()

    # run main
    main(args)

    pass