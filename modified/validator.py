import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from skimage import io
import skimage.color
import pandas as pd
from pathlib import Path

from deeplabcut.pose_estimation_tensorflow.nnet.net_factory import pose_net
from modified import pose_net
from deeplabcut.pose_estimation_tensorflow.dataset.pose_dataset import data_to_input
from deeplabcut.pose_estimation_tensorflow.nnet import predict as ptf_predict
from deeplabcut.pose_estimation_tensorflow import evaluate

class Validator:
    def __init__(self, cfg, pose_cfg, pose_cfg_path):
        stats_path = Path(pose_cfg_path).with_name('run-p{}-gs{}l{}h{}-c{}-ref{}.csv'.format(cfg['pcutoff'],
                                                                                       pose_cfg['global_scale'],
                                                                                       pose_cfg['scale_jitter_lo'],
                                                                                       pose_cfg['scale_jitter_up'],
                                                                                       pose_cfg['cropratio'],
                                                                                       pose_cfg['location_refinement']))
        self.lrf = open(str(stats_path), 'w')

        self.cfg = cfg
        self.pose_cfg = pose_cfg
        (self.Data, self.comparisonbodyparts, self.trainIndices, self.validIndices) = self.get_validation()
        self.inputs = tf.placeholder(tf.float32, shape=[pose_cfg.batch_size, None, None, 3])
        self.net_heads = (pose_net.PoseNet(pose_cfg)).test(self.inputs)
        self.outputs = [self.net_heads['part_prob']]
        if pose_cfg.location_refinement:
            self.outputs.append(self.net_heads['locref'])
        Numimages = len(self.Data.index)
        self.PredictedData = np.zeros((Numimages, 3 * len(pose_cfg['all_joints_names'])))

    def get_validation(self):
        import pandas as pd
        from modified import auxiliaryfunctions
        trainingsetfolder = auxiliaryfunctions.GetTrainingSetFolder(self.cfg)

        Data = pd.read_hdf(
            os.path.join(self.cfg["project_path"], str(trainingsetfolder), 'CollectedData_' + self.cfg["scorer"] + '.h5'),
            'df_with_missing')
        # Get list of body parts to evaluate network for
        comparisonbodyparts = auxiliaryfunctions.IntersectionofBodyPartsandOnesGivenbyUser(self.cfg, 'all')

        data, trainIndices, validIndices, testIndices, trainFraction = auxiliaryfunctions.LoadMetadata(
            os.path.join(self.cfg["project_path"], self.pose_cfg['metadataset']))

        return Data, comparisonbodyparts, trainIndices, validIndices


    def validate(self, sess, trainingsiterations):
        final_result = []
        # TODO:: Adapt to training/validation loss value
        for imageindex, imagename in tqdm(enumerate(self.Data.index)):
            image = io.imread(os.path.join(self.cfg['project_path'], imagename), mode='RGB')
            image = skimage.color.gray2rgb(image)
            image_batch = data_to_input(image)

            # Compute prediction with the CNN
            #[loss_val, summary] = sess.run([total_loss, merged_summaries],
            #                               feed_dict={inputs: image_batch})
            outputs_np = sess.run(self.outputs, feed_dict={self.inputs: image_batch})
            scmap, locref = ptf_predict.extract_cnn_output(outputs_np, self.pose_cfg)

            # Extract maximum scoring location from the heatmap, assume 1 person
            pose = ptf_predict.argmax_pose_predict(scmap, locref, self.pose_cfg.stride)
            self.PredictedData[imageindex,:] = pose.flatten()  # NOTE: thereby     cfg_test['all_joints_names'] should be same order as bodyparts!
        DLCscorer = 'Predictor'
        index = pd.MultiIndex.from_product(
        [[DLCscorer], self.pose_cfg['all_joints_names'], ['x', 'y', 'likelihood']],
        names=['scorer', 'bodyparts', 'coords'])

        # Saving results
        DataMachine = pd.DataFrame(self.PredictedData, columns=index, index=self.Data.index.values)
        #DataMachine.to_hdf(resultsfilename,'df_with_missing',format='table',mode='w')

        print("Validated validation set")
        DataCombined = pd.concat([self.Data.T, DataMachine.T], axis=0).T
        RMSE,RMSEpcutoff = evaluate.pairwisedistances(DataCombined, self.cfg["scorer"], DLCscorer, self.cfg["pcutoff"], self.comparisonbodyparts)
        validerror = np.nanmean(RMSE.iloc[self.validIndices].values.flatten())
        trainerror = np.nanmean(RMSE.iloc[self.trainIndices].values.flatten())
        validerrorpcutoff = np.nanmean(RMSEpcutoff.iloc[self.validIndices].values.flatten())
        trainerrorpcutoff = np.nanmean(RMSEpcutoff.iloc[self.trainIndices].values.flatten())
        results = [trainingsiterations,np.round(trainerror,2),np.round(validerror,2),self.cfg["pcutoff"],np.round(trainerrorpcutoff,2), np.round(validerrorpcutoff,2)]
        final_result.append(results)

        print("Results for",trainingsiterations," training iterations:", "train error:",np.round(trainerror,2), "pixels. Validation error:", np.round(validerror,2)," pixels.")
        print("With pcutoff of", self.cfg["pcutoff"]," train error:",np.round(trainerrorpcutoff,2), "pixels. Validation error:", np.round(validerrorpcutoff,2), "pixels")
        print("Thereby, the errors are given by the average distances between the labels by DLC and the scorer.")

        # Write to file
        self.lrf.write("{}, {:.5f}, {:.5f}, {:.5f}, {:.5f}\n".format(trainingsiterations, trainerror, trainerrorpcutoff, validerror, validerrorpcutoff))
        self.lrf.flush()

        return validerror

    def close(self):
        self.lrf.close()
