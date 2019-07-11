import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob

from sklearn.externals import joblib
from keras.models import load_model
import keras.backend as K

import utils as utl

import seaborn as sns
sns.set(context='talk', font_scale=0.75)
sns.set_style('white')

features = ['ZS1.LSS2.ANODE:DO_PPM',
            'ZS2.LSS2.ANODE:DO_PPM', 'ZS2.LSS2.ANODE:UP_PPM',
            'ZS3.LSS2.ANODE:DO_PPM', 'ZS3.LSS2.ANODE:UP_PPM',
            'ZS4.LSS2.ANODE:DO_PPM', 'ZS4.LSS2.ANODE:UP_PPM',
            'ZS5.LSS2.ANODE:DO_PPM', 'ZS5.LSS2.ANODE:UP_PPM',
            'ZS.LSS2.GIRDER:DO_PPM']

targets = ['SPS.BLM.21636.ZS1:LOSS_CYCLE_NORM',
           'SPS.BLM.21652.ZS2:LOSS_CYCLE_NORM',
           'SPS.BLM.21658.ZS3:LOSS_CYCLE_NORM',
           'SPS.BLM.21674.ZS4:LOSS_CYCLE_NORM',
           'SPS.BLM.21680.ZS5:LOSS_CYCLE_NORM']

"""
# (I) Plot scan in number of nodes / NN architecture
plot_targets = np.array(targets).take([0, 1, 2, 3, 4])

path_list = sorted(glob.glob('output/006_*/'))[:2]
n_paths = len(path_list)
blues = plt.get_cmap('Blues')
greens = plt.get_cmap('Greens')
fig_axs = None

for i, path in enumerate(path_list):
    scaler_in = joblib.load(path + '/scaler_in.save')
    scaler_out = joblib.load(path + '/scaler_out.save')
    loss_model = load_model(path + '/NN_model')
    sess = K.get_session()

    colors = (blues(float(i+1)/n_paths),
              greens(float(i+1)/n_paths))
    label = '1 hl, {:d} nodes'.format(
        int(path.split('_')[-1].split('/')[0]))
    fig_axs = utl.orthogonal_feature_scans(
        loss_model, scaler_in, scaler_out, features, targets,
        plot_targets, fig_axs=fig_axs, colors=colors, label=label)

plt.subplots_adjust(bottom=0.1, left=0.08, top=0.92, right=0.88,
                    hspace=0.23)
fig_axs[1][0, -1].legend(loc='upper left', bbox_to_anchor=(1.01, 1.1))
fig_axs[1][-1, -1].legend(loc='lower left', bbox_to_anchor=(1.01, -0.1))
# plt.savefig('FeatureScans_indivBLMs_diffNnodes_1HL_withGirderDO.pdf')
plt.show()
"""

# (II) Try a scan in girder position and see how response of different
# anode wires changes (do they just shift wrt. 0?)
# Use fixed NN model for now
plot_targets = np.array(targets).take([0, 1, 2, 3, 4])
girder_scan = np.linspace(-1.5, 1.5, 3)
n_pos = len(girder_scan)

path = 'output/006_1HL_noDO_32/'
scaler_in = joblib.load(path + '/scaler_in.save')
scaler_out = joblib.load(path + '/scaler_out.save')
loss_model = load_model(path + '/NN_model')
sess = K.get_session()

blues = plt.get_cmap('Blues')
greens = plt.get_cmap('Greens')
fig_axs = None
for i, pos in enumerate(girder_scan):
    girder_pos = {'DO': 42.8 + pos,
                  'UP': 68.1}  # use "default" for girder UP
    colors = (blues(float(i+3)/(n_pos+2)),
              greens(float(i+3)/(n_pos+2)))
    label = r'$\Delta x = {:.1f}$'.format(pos)
    fig_axs = utl.orthogonal_feature_scans(
        loss_model, scaler_in, scaler_out, features, targets,
        plot_targets, girder_pos=girder_pos, fig_axs=fig_axs,
        colors=colors, label=label)

plt.subplots_adjust(bottom=0.1, left=0.08, top=0.92, right=0.88,
                    hspace=0.23)
# Adjust axis limits: I don't want to include it in the utils.py module
[ax.set_ylim(0., 1.5e-13) for ax in fig_axs[1][:-1, :].flatten()]
[ax.set_ylim(0., 3e-13) for ax in fig_axs[1][-1, :]]

# Legends
fig_axs[1][0, -1].legend(loc='upper left', bbox_to_anchor=(1.01, 1.1))
fig_axs[1][-1, -1].legend(loc='lower left', bbox_to_anchor=(1.01, -0.1))
plt.savefig('FeatureScans_indivBLMs_32nodes_1HL_movingGirderDO.pdf')
plt.show()