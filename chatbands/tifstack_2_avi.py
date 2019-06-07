import tiffcapture as tc
import cv2
import numpy as np


def data_2_avi(data, filename, fps=15):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('{}.avi'.format(filename), fourcc, fps, np.shape(data)[1:3][::-1])

    for img in data:
        #out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        out.write(np.uint8(img))
    cv2.destroyAllWindows()

def data_2_mp4(data, filename, fps=15):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('{}.mp4'.format(filename), fourcc, fps, np.shape(data)[1:3][::-1])
    for img in data:
        #out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        exf_color = cv2.cvtColor(np.uint8(img), cv2.COLOR_GRAY2BGR)
        out.write(exf_color)
    cv2.destroyAllWindows()

def load_data(filename):
    tiff = tc.opentiff(filename)  # open img
    _, first_img = tiff.retrieve()
    video = np.zeros((tiff.length, *tiff.shape))
    print(video.shape)
    for i, img in enumerate(tiff):
        video[i] = img
    max_pix = video.max()
    if max_pix != 0:
        video /= max_pix # TODO:: check max on maybe a specific set of axes
    return video


def switch_axis(data):
    return np.swapaxes(data, 0, 1)


def play_video(video, speed=20):
    cv2.namedWindow('video')
    for img in video:
        cv2.imshow('video', img)
        cv2.waitKey(speed)
    cv2.destroyWindow('video')


def get_video(filename):
    data = load_data(filename)
    print('Normal axis')
    play_video(data, 5)
    #disp_images(data[:][::20], cols=9, cmap='gray')
    print('Modified axis')
    mod_data = switch_axis(data)
    play_video(mod_data, 5)
    return mod_data

def disp_images(imgs, txts=None, cols=10, title='', cmap=None):
    from matplotlib import pyplot as plt
    if txts is None:
        txts = ['']*len(imgs)
    if len(imgs) <= cols:
        f, axarr = plt.subplots(1, len(imgs), dpi=200, sharex='all')
        for i, (img, txt) in enumerate(zip(imgs, txts)):
            axarr[i % cols].imshow(img, cmap=cmap)
            axarr[i % cols].axis('off')
            axarr[i % cols].set_title(txt)
        f.suptitle(title, fontsize=16)
        #plt.show()
    elif cols == 1:
        f, axarr = plt.subplots(len(imgs), 1, dpi=200, sharex='all')
        for i, (img, txt) in enumerate(zip(imgs, txts)):
            axarr[i].imshow(img, cmap=cmap)
            axarr[i].axis('off')
            axarr[i].set_title(txt)
        f.suptitle(title, fontsize=16)
        #plt.show()
    else:
        f, axarr = plt.subplots(len(imgs) // cols, cols, figsize=(15, 4), dpi=200, sharex='all')
        for i, (img, txt) in enumerate(zip(imgs, txts)):
            axarr[i // cols][i % cols].imshow(img, cmap=cmap)
            axarr[i // cols][i % cols].axis('off')
            axarr[i // cols][i % cols].set_title(txt)
        f.suptitle(title, fontsize=16)
        #plt.show()
    plt.savefig("chat_orig", bbox_inches='tight', pad_inches=0)
    plt.close()

#filename = '00629_2L_C08_chAT_STD.tif'
#get_video(filename)
