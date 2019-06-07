import cv2
import os
import tiffstack2avi
import glob


path = r"C:\Users\bramn\Documents\MAI\Thesis\videos"
print(path)
#tiffstack2avi.convert()
def rename():
    for i in range(662, 663):
        os.rename(path + "\\20181030_run000_00000" + str(i).zfill(3) + ".tif", path + "\\frame"+str(i)+".tif")
tiffstack2avi.convert()

# image_folder = path
# video_name = 'video_xvid.avi'
#
# images = [img for img in os.listdir(image_folder) if img.endswith(".tif")]
# frame = cv2.imread(os.path.join(image_folder, images[0]))
# height, width, layers = frame.shape
#
# #video = cv2.VideoWriter(video_name, 0, 1, (width, height))
#
# video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), 1, (width, height))
#
# for image in images:
#     video.write(cv2.imread(os.path.join(image_folder, image)))
#
# cv2.destroyAllWindows()
# video.release()