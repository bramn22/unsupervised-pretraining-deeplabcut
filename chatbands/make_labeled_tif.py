import os
from os.path import split, splitext
import pandas as pd
from chatbands import tifstack_2_avi
import cv2
#from tifffile import imsave

def create_labeled_tif(tif_path, data_path, dest_path, step_size=20):

    tif_dir, tif = split(tif_path)
    video = tifstack_2_avi.get_video(tif_path)*255
    length, height, width = video.shape
    # tifstack_2_avi.play_video(video, 5)
    # tif_name, ext = splitext(tif)
    # data_dir, data = split(data_path)
    # data_file = [f for f in os.listdir(path) if f.startswith(name) and f.endswith('.h5')][0]

    df = pd.read_hdf(data_path)

    pos = 0
    frame = 0
    prev_ON = None
    prev_OFF = None
    for i, row in df.iterrows():
        if row[:, 'ON', 'likelihood'].values[0] >= 0.0:
            ON = row[:, 'ON', 'y']
            cv2.circle(video[frame], (pos, ON), 3, (255, 0, 0), -1)
            if prev_ON is not None:
                cv2.line(video[frame], prev_ON, (pos, ON), (255, 0, 0), thickness=2)
            prev_ON = (pos, ON)
        if row[:, 'OFF', 'likelihood'].values[0] >= 0.0:
            OFF = row[:, 'OFF', 'y']
            cv2.circle(video[frame], (pos, OFF), 3, (255, 120, 0), -1)

            if prev_OFF is not None:
                cv2.line(video[frame], prev_OFF, (pos, OFF), (255, 120, 0), thickness=2)
            prev_OFF = (pos, OFF)
        pos += step_size
        if pos > width:
            frame += 1
            pos = 0
            prev_ON = None
            prev_OFF = None

    # tifstack_2_avi.play_video(video, 15)
    tifstack_2_avi.data_2_mp4(video, 'resulttif_{}'.format(tif), 60)
    # TODO: save as a tif file!!
    # image = np.zeros((32, 256, 256), 'uint16')
    # imsave('multipage.tif', image)
