import os
import cv2
import numpy as np

from chatbands import tifstack_2_avi
from deeplabcut import auxiliaryfunctions


def extract_video(chatbands, chatbands_path, config_path, step_size=20, side_width=100):
    # TODO: place video in correct folder
    cfg = auxiliaryfunctions.read_config(config_path)
    direc = os.path.join(cfg['project_path'], 'videos')


    for i, c in enumerate(chatbands):
        chat = os.path.join(chatbands_path, c)
        filename = chat + '_chAT_STD.tif'
        video = tifstack_2_avi.get_video(filename)
        length, height, width = video.shape

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(os.path.join(direc, 'video{}.mp4'.format(c)), fourcc, 30, (side_width*2+1, height))
        #out = cv2.VideoWriter('video.mp4', fourcc, 30., (height, width))

        for i, img in enumerate(video):
            for x in range(0, np.shape(img)[1], step_size):
                if x - side_width < 0:
                    right = img[:, x + 1:x + side_width + 1] * 255
                    left = np.flip(right, axis=1)
                    exf = np.hstack((left, img[:, x:x + 1] * 255, right))
                elif x + side_width > width:
                    left = img[:, x - side_width:x] * 255
                    right = np.flip(left, axis=1)
                    exf = np.hstack((left, img[:, x:x + 1] * 255, right))
                else:
                    exf = img[:, x - side_width:x + side_width + 1] * 255  # TODO:: check correctness!!
                exf_color = cv2.cvtColor(np.uint8(exf), cv2.COLOR_GRAY2BGR)
                out.write(exf_color)

        cv2.destroyAllWindows()




