from chatbands import extract_labeled_data
from chatbands import extract_unlabeled_data
from pretraining.pretrain_new import create_new_project
from modified.trainingsetmanipulation import create_training_dataset as pose_training_set
from modified import training
from modified import evaluate
from chatbands import predict_chAT
from chatbands import make_labeled_tif
import os


# print(create_new_project('chATbands','NERF','','',copy_videos=False, videotype='.avi'))
config_path = r'C:\Users\bramn\Documents\NERF\behavioral-tracking\deeplabcut\chATbands-NERF-2019-05-22\config.yaml'

chatbands_path = r'C:\Users\bramn\Documents\NERF\chatbands_data'
chatbands = ['00629_2R_C01']

# extract_labeled_data.extract_and_label(chatbands, chatbands_path, config_path)
# pose_training_set(config_path, 1)
# training.train_network(config_path, max_snapshots_to_keep=10, gputouse=0, maxiters=700000)
# evaluate.evaluate_network(config_path, plotting=True)
# extract_unlabeled_data.extract_video(chatbands, chatbands_path, config_path, step_size=50)
dest_path = r'C:\Users\bramn\Documents\NERF\behavioral-tracking\deeplabcut\chATbands-NERF-2019-05-22\videos'
# predict_chAT.analyze_videos(config_path, [os.path.join(dest_path, 'video00629_2R_C01_10.mp4')], 'mp4', save_as_csv=True, destfolder=dest_path)
make_labeled_tif.create_labeled_tif(os.path.join(chatbands_path, "00629_2R_C01_chAT_STD.tif"), os.path.join(dest_path, "video00629_2R_C01_50DeepCut_resnet50_chATbandsMay22shuffle1_71000.h5"), chatbands_path, step_size=50)

