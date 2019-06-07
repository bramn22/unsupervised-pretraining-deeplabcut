import deeplabcut

from pretraining.pretrain_new import create_new_project
from pretraining.unsup_frame_extraction import extract_frames as pre_extract
from modified import training
from modified import evaluate
from modified.trainingsetmanipulation import create_training_dataset as pose_training_set
from pretraining.unsup_trainingsetmanipulation import create_training_dataset as unsup_training_set
from pretraining.unsup_training import unsup_train_network
config_path = r'C:\Users\bramn\Documents\NERF\behavioral-tracking\deeplabcut\MouseTracking-NERF-2019-05-07\config.yaml'
# pre_config_path = r'C:\Users\bramn\Documents\NERF\behavioral-tracking\deeplabcut\OpenField50Context-NERF-2019-02-03\pretrain_config.yaml'


# print(create_new_project('OpenField50Context','NERF',r'C:\Users\bramn\Documents\NERF\videos',r'C:\Users\bramn\Documents\NERF\videos',copy_videos=False, videotype='.avi'))
# pre_extract(pre_config_path, 'uniform', userfeedback=False)
# unsup_training_set(pre_config_path, 1)
# unsup_train_network(pre_config_path, shuffle=1, maxiters=1000000, max_snapshots_to_keep=11, gputouse=0)


#deeplabcut.extract_frames(config_path)
#deeplabcut.label_frames(config_path)
#deeplabcut.check_labels(config_path)

# pose_training_set(config_path, 1)
# training.train_network(config_path, max_snapshots_to_keep=10, gputouse=0, maxiters=700000)
evaluate.evaluate_network(config_path, plotting=True, gputouse=0)
#training.train_network(config_path, gputouse=0, maxiters=50000)