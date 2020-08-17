import cv2
import datetime

def gstreamer_pipeline(
        capture_width=1280,
        capture_height=720,
        display_width=1280,
        display_height=720,
        framerate=60,
        flip_method=0,
        ):
        return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                capture_width,
                capture_height,
                framerate,
                flip_method,
            )
        )

class camera:
    def __init__(self, grayscale: bool=False):        
        """
        This woking with Pi camera.

        Parameters
        ----------
        grayscale : bool, optional
            This value decide whether tranform images to grayscale.
            The default is False.

        Returns
        -------
        None.

        """
        self.grayscale = grayscale
        self.cam = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        if self.cam.isOpened():
            print("Camera is ready")


    def take_a_pic(self):
        _, image = self.cam.read()
        if self.grayscale:
            image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
        return image

    
    def save_pic(self, image):    
        """
        save the images.

        Parameters
        ----------
        image : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """        
        cv2.imwrite('./images/camera/{}.png'.format(datetime.datetime.now()), image)
        print("Finish")
        
