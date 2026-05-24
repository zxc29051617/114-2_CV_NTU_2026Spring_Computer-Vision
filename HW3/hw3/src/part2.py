import numpy as np
import cv2
from cv2 import aruco
from tqdm import tqdm
from utils import solve_homography, warping


def planarAR(REF_IMAGE_PATH, VIDEO_PATH):
    """
    Reuse the previously written function "solve_homography" and "warping" to implement this task
    :param REF_IMAGE_PATH: path/to/reference/image
    :param VIDEO_PATH: path/to/input/seq0.avi
    """
    video = cv2.VideoCapture(VIDEO_PATH)
    ref_image = cv2.imread(REF_IMAGE_PATH)
    h, w, c = ref_image.shape
    film_h, film_w = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    film_fps = video.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    videowriter = cv2.VideoWriter("output2.avi", fourcc, film_fps, (film_w, film_h))
    # arucoDict = aruco.Dictionary_get(aruco.DICT_4X4_50)
    # arucoParameters = aruco.DetectorParameters_create()
    arucoDict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    arucoParameters = aruco.DetectorParameters()
    ref_corns = np.array([[0, 0], [w, 0], [w, h], [0, h]])

    detector = aruco.ArucoDetector(arucoDict, arucoParameters) if hasattr(aruco, 'ArucoDetector') else None
    pbar = tqdm(total = 353)
    while (video.isOpened()):
        ret, frame = video.read()
        if ret:  ## check whethere the frame is legal, i.e., there still exists a frame
            if detector is not None:
                corners, ids, _ = detector.detectMarkers(frame)
            else:
                corners, ids, _ = aruco.detectMarkers(frame, arucoDict, parameters=arucoParameters)

            if ids is not None and len(corners) > 0:
                marker_corns = max(
                    (corner.reshape(4, 2) for corner in corners),
                    key=cv2.contourArea
                )
                H = solve_homography(ref_corns, marker_corns)

                xmin = max(0, int(np.floor(marker_corns[:, 0].min())))
                xmax = min(film_w, int(np.ceil(marker_corns[:, 0].max())) + 1)
                ymin = max(0, int(np.floor(marker_corns[:, 1].min())))
                ymax = min(film_h, int(np.ceil(marker_corns[:, 1].max())) + 1)

                if xmin < xmax and ymin < ymax:
                    frame = warping(ref_image, frame, H, ymin, ymax, xmin, xmax, direction='b')

            videowriter.write(frame)
            pbar.update(1)

        else:
            break

    pbar.close()
    video.release()
    videowriter.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    # ================== Part 2: Marker-based planar AR========================
    VIDEO_PATH = '../resource/seq0.mp4'
    # TODO: you can change the reference image to whatever you want
    REF_IMAGE_PATH = '../resource/hehe.jpg' 
    planarAR(REF_IMAGE_PATH, VIDEO_PATH)
