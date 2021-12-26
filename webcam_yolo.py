import cv2
from imageai import Detection


MODEL_PATH = "yolo.h5"
FRAME_WIDTH = 1300
FRAME_HEIGHT = 1500
# Only objects for which the algorithm is more confidence than this percentage will show
MIN_PROB = 50
DETECTION_SPEED = "flash"
WINDOW_NAME = "YOLO"
EXIT_KEY = 27


def load_detector():
    """
    :return: an object detector
    """
    detector = Detection.ObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(MODEL_PATH)
    detector.loadModel(detection_speed=DETECTION_SPEED)
    return detector


def set_cam():
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return cam


def one_frame_detector(in_img, detector):
    """
    detect objects shows in the input image
    :param in_img: the image to analyse
    :param detector: the algorithm to use
    :return: the image with marked object
    """
    img, detections = detector.detectObjectsFromImage(in_img,
                                                      input_type="array",
                                                      minimum_percentage_probability=MIN_PROB,
                                                      output_type="array")
    return img


def run_webcam_detector():
    detector = load_detector()
    cam = set_cam()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.startWindowThread()
    while True:
        ret, frame = cam.read()
        out_frame = one_frame_detector(frame, detector)
        cv2.imshow(WINDOW_NAME, out_frame)
        key_code = cv2.waitKey(1)

        if (cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1) or (key_code == EXIT_KEY):
            break
    cam.release()
    cv2.destroyAllWindows()


def main():
    run_webcam_detector()


if __name__ == '__main__':
    main()
