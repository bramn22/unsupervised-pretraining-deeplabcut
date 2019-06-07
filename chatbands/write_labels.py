import h5py
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd

#filename = 'data.h5'
#df = pd.read_hdf(filename)

class LabelWriter:
     def __init__(self, cfg, direc, img_prefix):
          # TODO:: create dataframe from csv -> save dataframe as csv and hdf
          self.direc = direc
          self.scorer = cfg['scorer']
          bodyparts = cfg['bodyparts']
          videos = cfg['video_sets'].keys()
          markerSize = cfg['dotsize']
          alpha = cfg['alphavalue']
          colormap = plt.get_cmap(cfg['colormap'])
          project_path = cfg['project_path']
          # index =np.sort([fn for fn in glob.glob(os.path.join(self.dir,'*.png')) if ('labeled.png' not in fn)])

          newimages = [os.path.join(img_prefix, f) for f in os.listdir(self.direc) if f.endswith('.' + 'png')]  # imagenames
          self.df = None
          a = np.empty((len(newimages), 2,))
          a[:] = np.nan
          for bodypart in bodyparts:
               index = pd.MultiIndex.from_product([[self.scorer], [bodypart], ['x', 'y']],
                                                  names=['scorer', 'bodyparts', 'coords'])
               frame = pd.DataFrame(a, columns=index, index=newimages)
               self.df = pd.concat([self.df, frame], axis=1)

     def save_labels(self, labels, part):
          for (i, x, y) in labels:
               self.df.loc[i][self.scorer, part, 'x'] = x
               self.df.loc[i][self.scorer, part, 'y'] = y


     def write_to_file(self):
          self.df.sort_index(inplace=True)
          self.df.to_csv(os.path.join(self.direc, "CollectedData_" + self.scorer + ".csv"))
          self.df.to_hdf(os.path.join(self.direc,"CollectedData_" + self.scorer + '.h5'),'df_with_missing',format='table', mode='w')


     # print('Working on folder: {}'.format(os.path.split(str(self.dir))[-1]))
     #self.df.loc[self.relativeimagenames[self.iter]][scorer, bp[0][-2], 'y'] = bp[-1][1]
