import pandas as pd
from chatbands import tifstack_2_avi
from chatbands import write_labels
import cv2
import numpy as np
import os
from deeplabcut import auxiliaryfunctions


def extract_frames(data, video, direc, img_prefix, side_width=100, idx_start=0):
    length, height, width = video.shape
    labels = []
    for i, (x, y, s) in data.iterrows():
        # cv2.circle(video[s], (x, y), 5, 1, -1) # TODO:: turn off!!
        # Extract frames
        if x-side_width < 0:
            right = video[s, :, x + 1:x + side_width + 1] * 255
            left = np.flip(right, axis=1)
            exf = np.hstack((left, video[s, :, x:x+1]*255, right))
        elif x+side_width > width:
            left = video[s, :, x-side_width:x] * 255
            right = np.flip(left, axis=1)
            exf = np.hstack((left, video[s, :, x:x+1]*255, right))
        else:
            exf = video[s, :, x-side_width:x+side_width+1]*255 # TODO:: check correctness!!
        cv2.imwrite(os.path.join(direc, ('img{:03d}.png'.format(i+idx_start))), exf)
        labels.append((os.path.join(img_prefix, ('img{:03d}.png'.format(i+idx_start))), side_width, y))
    return labels


def extract_and_label(chatbands, chatbands_path, config_path):
    for c in chatbands:
        chat = os.path.join(chatbands_path, c)
        df_ON = pd.read_csv(chat+'_ON.xls', sep='\t', index_col=0)
        df_OFF = pd.read_csv(chat+'_OFF.xls', sep='\t', index_col=0)

        df_ON = df_ON[['X', 'Y', 'Slice']]
        df_OFF = df_OFF[['X', 'Y', 'Slice']]

        df_ON['Slice'] -= 1
        df_OFF['Slice'] -= 1

        filename = chat+'_chAT_STD.tif'
        video = tifstack_2_avi.get_video(filename)
        #tifstack_2_avi.play_video(video, 80)
        cfg = auxiliaryfunctions.read_config(config_path)

        direc = os.path.join(cfg['project_path'], 'labeled-data', c)
        img_prefix = os.path.join('labeled-data', c)
        os.mkdir(direc)
        print('Created "{}"'.format(direc))
        labels_ON = extract_frames(df_ON, video, direc, img_prefix, idx_start=0)
        labels_OFF = extract_frames(df_OFF, video, direc, img_prefix, idx_start=len(labels_ON))
        print(np.shape(labels_ON), np.shape(labels_OFF))
        df = write_labels.LabelWriter(cfg, direc, img_prefix)
        df.save_labels(labels_ON, 'ON')
        df.save_labels(labels_OFF, 'OFF')
        df.write_to_file()

        #cfg['video_sets'] = ...  # TODO
        cfg['bodyparts'] = ['ON', 'OFF']
        auxiliaryfunctions.write_config(config_path, cfg)
        cv2.destroyAllWindows()
# for i, (x, y, s) in df_ON.iterrows():
#     cv2.circle(video[s], (x, y), 5, 255, -1)
#     cv2.imshow('video', video[s])
#     cv2.waitKey(20)
#
# for i, (x, y, s) in df_OFF.iterrows():
#     cv2.circle(video[s], (x, y), 5, 255, -1)
#     cv2.imshow('video', video[s])
#     cv2.waitKey(20)

#tifstack_2_avi.play_video(video, 80)