
import os
from pathlib import Path


def unsup_train_network(config,shuffle=1,trainingsetindex=0,gputouse=None,max_snapshots_to_keep=5,autotune=False,displayiters=None,saveiters=None,maxiters=None):
    import tensorflow as tf
    import pretraining.unsup_train as unsup_train
    from deeplabcut.utils import auxiliaryfunctions

    tf.reset_default_graph()
    start_path = os.getcwd()

    # Read file path for pose_config file. >> pass it on
    cfg = auxiliaryfunctions.read_config(config)
    modelfoldername = Path('pretrain/dlc-models/trainset')
    #modelfoldername = auxiliaryfunctions.GetModelFolder(cfg["TrainingFraction"][trainingsetindex], shuffle, cfg)
    poseconfigfile = Path(os.path.join(cfg['project_path'], str(modelfoldername), "train", "pretrain_pose_cfg.yaml"))
    if not poseconfigfile.is_file():
        print("The training datafile ", poseconfigfile, " is not present.")
        print("Probably, the training dataset for this secific shuffle index was not created.")
        print(
            "Try with a different shuffle/trainingsetfraction or use function 'create_training_dataset' to create a new trainingdataset with this shuffle index.")
    else:
        # Set environment variables
        if autotune is not False:  # see: https://github.com/tensorflow/tensorflow/issues/13317
            os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'
        if gputouse is not None:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gputouse)
        try:
            unsup_train.train(str(poseconfigfile), displayiters, saveiters, maxiters,
                              max_to_keep=max_snapshots_to_keep)  # pass on path and file name for pose_cfg.yaml!
        except BaseException as e:
            raise e
        finally:
            os.chdir(str(start_path))
        print(
            "The network is now trained and ready to evaluate. Use the function 'evaluate_network' to evaluate the network.")

