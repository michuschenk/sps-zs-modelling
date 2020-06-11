import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from shutil import copy2
import joblib as jbl
import glob
import pickle as pkl

from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, Dropout
from keras.optimizers import Adam
from keras.initializers import glorot_normal
from keras.callbacks import EarlyStopping
from keras.models import load_model
import keras.backend as K

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle
from sklearn.externals import joblib

import utils as utl
import data_io as dio

import seaborn as sns
sns.set(context='talk', font_scale=0.75)
sns.set_style('white')


# *****************
# (0) Basic config.
# Some network configurations that perform reasonably well
nn_topologies = {
    # '1HL_noDO': {'n_nodes_1': 7, 'n_nodes_2': 0, 'dropout': False},
    '2HL_wDO': {'n_nodes_1': 100, 'n_nodes_2': 100, 'dropout': True},
    '2HL_wDO_small': {'n_nodes_1': 20, 'n_nodes_2': 10, 'dropout': True},
    '1HL_noDO': {'n_nodes_1': 15, 'n_nodes_2': 0, 'dropout': False}
}
# training_params = {
#     'early_stopping': False, 'batch_size': 10, 'n_epochs': 1200,   # 1200,
#     'learning_rate': 5e-4
# }
training_params = {
    'early_stopping': True, 'batch_size': 10, 'n_epochs': 600,   # 1200,
    'learning_rate': 6e-4,
    # 'train_data': ['050418', '011118'],
    # 'test_data': '270318',
    # 'train_data': ['011118', '270318'],
    # 'test_data': '050418',
    'train_data': ['011118'],  # , '260418', '050418'],  # , '270318'],  #, '260418'],
    'test_data': '270318',
    'features':
        ['ZS1.LSS2.ANODE:DO_PPM',  # 'ZS1.LSS2.ANODE:UP_PPM',
         'ZS2.LSS2.ANODE:DO_PPM', 'ZS2.LSS2.ANODE:UP_PPM',
         'ZS3.LSS2.ANODE:DO_PPM', 'ZS3.LSS2.ANODE:UP_PPM',
         'ZS4.LSS2.ANODE:DO_PPM', 'ZS4.LSS2.ANODE:UP_PPM',
         'ZS5.LSS2.ANODE:DO_PPM', 'ZS5.LSS2.ANODE:UP_PPM'],
         # 'ZS.LSS2.GIRDER:DO_PPM'],  # 'ZS.LSS2.GIRDER:UP_PPM'],
    'targets':
        ['SPS.BLM.21636.ZS1:LOSS_CYCLE_NORM',
         'SPS.BLM.21652.ZS2:LOSS_CYCLE_NORM',
         'SPS.BLM.21658.ZS3:LOSS_CYCLE_NORM',
         'SPS.BLM.21674.ZS4:LOSS_CYCLE_NORM',
         'SPS.BLM.21680.ZS5:LOSS_CYCLE_NORM'],
         # 'SPS.BLM.21694.TCE:LOSS_CYCLE_NORM',
         # 'SPS.BLM.21772.TPST:LOSS_CYCLE_NORM']
    'train_total_loss': False
}

config = nn_topologies['2HL_wDO']
config.update(training_params)

plot_training_data = True
group_duplicates = False

# Place to save code, plots, etc., for reproducibility
# Change output_dir to avoid overwriting
# Find latest directory and check its config. If it's not the same
# anymore move on and create new directory.
latest_output = sorted(glob.glob('output/*/'))[-1]
dir_number = int(latest_output.split('output/')[-1].split('/')[0].split('_')[0])
with open(latest_output + '/config.pkl', 'rb') as fid:
    latest_config = pkl.load(fid)

retrain_NN = False
if latest_config != config:
    print('Configuration has changed. Create new output directory and ' +
          'retrain NN.')
    retrain_NN = True
    dir_number += 1
    output_dir = 'output/{:03d}/'.format(dir_number)
else:
    print('Same configuration, no need to retrain NN')
    output_dir = latest_output

Path(output_dir).mkdir(parents=True, exist_ok=True)
copy2('main.py', output_dir + 'main.py.state')
copy2('utils.py', output_dir + 'utils.py.state')

# Save config for future reference
with open(output_dir + '/config.pkl', 'wb') as fid:
    pkl.dump(config, fid)

# train_sets = {
#     THIS ONE WORKS PRETTY WELL IN GENERAL (INCL GIRDER)
#     'train_combi_1': {
#         'filepath': './timber_data/',
#         'filenames': ['TIMBER_DATA_011118_1300-1600.xls',
#                       'TIMBER_DATA_260418_1730-2300.xls'],
#         'start_times': ['2018-11-01 13:18:00.000',
#                         '2018-04-26 17:30:00.000'],
#         'end_times': ['2018-11-01 15:38:00.000',
#                       '2018-04-26 23:00:00.000'],
#         'comment': 'general purpose training set, with a bit of girder DO'
#     },
#     THIS ONE IS TO CHECK HOW THE GIRDER RESPONSE CHANGES WHEN INCL.
#     DATA FROM 05.04. (EXTENSIVE GIRDER SCANS).
#     'train_combi_2': {
#         'filepath': './timber_data/',
#         'filenames': ['TIMBER_DATA_011118_1300-1600.xls',
#                       'TIMBER_DATA_050418_2110-2315.xls'],
#         'start_times': ['2018-11-01 13:18:00.000',
#                         '2018-04-05 21:10:00.000'],
#         'end_times': ['2018-11-01 15:38:00.000',
#                       '2018-04-05 23:15:00.000'],
#         'comment': 'includes extensive girder scan'
#     }
# }

# train_cfg_1 = dio.Dataconfig(config=dio.config_dict['260418'])

# ********************
# (1) Data I/O
# Define features and targets
features = config['features']
targets = config['targets']
train_total_loss = config['train_total_loss']

# Load training data
train_cfgs = []
for td in config['train_data']:
    train_cfgs.append(dio.Dataconfig(config=dio.config_dict[td]))
train_cfgs = np.sum(train_cfgs)
train_data = dio.load_data(train_cfgs)
train_data = utl.filter_zs_blm_outliers(train_data, threshold=5e-15)

x_train, y_train = train_data[features], train_data[targets]

if plot_training_data:
    # Plot every dataset (i.e. from different days) separately
    for i in range(len(train_cfgs.config['filenames'])):
        t_start = np.datetime64(train_cfgs.config['start_times'][i])
        t_end = np.datetime64(train_cfgs.config['end_times'][i])
        mask_time = ((train_data['Timestamp (UTC_TIME)'] >= t_start) &
                     (train_data['Timestamp (UTC_TIME)'] <= t_end))
        sub_data = train_data[mask_time]
        # sub_data = utl.add_total_loss(sub_data, targets)
        # utl.plot_data(data=sub_data, title='Training data\n' +
        #                                    str(t_start).split('T')[0])
        # plt.savefig(output_dir + 'Training_data_set{:d}.pdf'.format(i))

        utl.plot_individual_blms(
            data=sub_data,
            title='Training data (indiv. BLMs)\n' +
                  str(t_start).split('T')[0])

        plt.savefig(output_dir + 'Training_data_set{:d}_individBLMs.pdf'
                    .format(i))
        plt.show()

# *****************************
# (2) Prepare data for training
# TODO: Use sklearn pipelines
if group_duplicates:
    x_train, y_train = utl.group_duplicates(
        x_train, y_train, abs_diff=0.001)

# Shuffle data and apply scaling to in- *and* outputs
x_train, y_train = shuffle(x_train, y_train, random_state=0)

scaler_in = MinMaxScaler(feature_range=(-0.5, 0.5))
x_train = scaler_in.fit_transform(x_train)
scaler_out = MinMaxScaler()
if train_total_loss:
    y_train = np.array(np.sum(y_train, axis=1))
    y_train = scaler_out.fit_transform(y_train.reshape(-1, 1))
else:
    y_train = scaler_out.fit_transform(y_train)

# Save scalers to reimport when reusing model
jbl.dump(scaler_in, output_dir + 'scaler_in.save')
jbl.dump(scaler_out, output_dir + 'scaler_out.save')


# **************************************
# (3) Define neural network architecture
n_features = x_train.shape[1]
n_targets = y_train.shape[1]

loss_model = Sequential()

# 1st hidden layer
loss_model.add(Dense(config['n_nodes_1'], input_shape=(n_features,),
                     kernel_initializer=glorot_normal(seed=0),
                     activation='linear', name='dense1_loss_model'))
loss_model.add(LeakyReLU(alpha=0.2))

# Dropout layer
if config['dropout']:
    loss_model.add(Dropout(rate=0.2, noise_shape=None, seed=None,
                           name='dropout1_loss_model'))

# 2nd hidden layer
if config['n_nodes_2']:
    loss_model.add(Dense(config['n_nodes_2'],
                         kernel_initializer=glorot_normal(seed=2),
                         activation='linear', name='dense2_loss_model'))
    loss_model.add(LeakyReLU(alpha=0.2))

# Output layer
loss_model.add(
    Dense(n_targets, kernel_initializer=glorot_normal(seed=2),
          activation='linear', name='layerout_loss_model'))

# Optimiser and loss
adam_opt = Adam(lr=config['learning_rate'], beta_1=0.9, beta_2=0.999,
                epsilon=None, amsgrad=False)
loss_model.compile(optimizer=adam_opt, loss='mean_squared_error',
                   metrics=['acc'])

# Training
callbacks = []
if config['early_stopping']:
    callbacks.append(EarlyStopping(patience=50))

if retrain_NN:
    training_history = loss_model.fit(
        x_train, y_train, validation_split=0.12,
        epochs=config['n_epochs'],
        callbacks=callbacks,
        batch_size=config['batch_size'])

    # Visualise training evolution
    training_history = pd.DataFrame(data=training_history.history)
    utl.plot_training_evolution(training_history)
    plt.savefig(output_dir + 'training_history.pdf')
    plt.show()
    loss_model.save(output_dir + 'NN_model')
else:
    scaler_in = joblib.load(output_dir + '/scaler_in.save')
    scaler_out = joblib.load(output_dir + '/scaler_out.save')
    loss_model = load_model(output_dir + '/NN_model')
    sess = K.get_session()

# ********************************
# (4) Make predictions on test set
# test_sets = {
#     '050418': {
#         'filepath': './timber_data/',
#         'filenames': ['TIMBER_DATA_050418_2110-2315.xls'],
#         'start_times': ['2018-04-05 21:10:00.000'],
#         'end_times': ['2018-04-05 23:15:00.000']
#     },
#     '300318': {
#         'filepath': './timber_data/',
#         'filenames': ['TIMBER_DATA_300318_1500-2130.xls'],
#         'start_times': ['2018-03-30 15:00:00.000'],
#         'end_times': ['2018-03-30 21:30:00.000']
#     },
#     '220918': {
#         'filepath': './timber_data/',
#         'filenames': ['TIMBER_DATA_220918_1235-1305_GIRDER.xls'],
#         'start_times': ['2018-09-22 12:35:00.000'],
#         'end_times': ['2018-09-22 13:05:00.000']
#     },
#     '270318': {
#         'filepath': './timber_data/',
#         'filenames': ['TIMBER_DATA_270318_1745-280318_0130.xls'],
#         'start_times': ['2018-03-27 19:20:00.000'],
#         'end_times': ['2018-03-28 01:30:00.000'],
#     }
# }

test_cfg = dio.Dataconfig(config=dio.config_dict[config['test_data']])
test_data = dio.load_data(test_cfg)
test_data = utl.filter_zs_blm_outliers(test_data, threshold=5e-15)
x_test, y_test = test_data[features], test_data[targets]

# ***************
# (4a) Total loss
test_data = utl.add_total_loss(test_data, targets)
t_start = np.datetime64(test_cfg.config['start_times'][0])
_, axs = utl.plot_data(
    test_data, title='Test data\n' + str(t_start).split('T')[0])

x_test = scaler_in.transform(x_test)
pred_test = loss_model.predict(x_test)
pred_test = scaler_out.inverse_transform(pred_test)
pred_test_tot = np.sum(pred_test, axis=1)
y_test_tot = np.array(np.sum(y_test, axis=1))

axs[-1].plot(test_data['Timestamp (UTC_TIME)'],
             pred_test_tot, c='dodgerblue',
             label='Model\nprediction')
axs[-1].set_ylim(0, 4e-13)
axs[-1].legend(loc='upper left', bbox_to_anchor=(1., 1.05))
plt.savefig(output_dir + 'TestData_withPrediction.pdf')
plt.show()


if not train_total_loss:
    # ********************
    # (4b) Individual BLMs
    # Reload test set
    test_data = dio.load_data(test_cfg)
    test_data = utl.filter_zs_blm_outliers(test_data, threshold=5e-15)
    test_data = utl.add_total_loss(test_data, targets)
    x_test, y_test = test_data[features], test_data[targets]

    # Add predictions
    x_test = scaler_in.transform(x_test)
    pred_test = loss_model.predict(x_test)
    pred_test = scaler_out.inverse_transform(pred_test)
    cols_pred = [i + '_PRED' for i in targets]
    pred_test = pd.DataFrame(data=pred_test, columns=cols_pred)

    test_data = pd.concat([test_data, pred_test], axis=1)
    plot_targets = np.array(targets).take([0, 1, 2, 3, 4])
    _, axs = utl.plot_individual_blms_predictions(
        data=test_data,
        title='Test data (indiv. BLMs)\n' + str(t_start).split('T')[0],
        plot_features=features,
        plot_targets=plot_targets)

    plt.savefig(output_dir + 'TestData_withPrediction_individBLMs.pdf')
    plt.show()

# *********************
# (4c) Orthogonal scans
# Perform sanity checks: fake scans for all anodes
plot_targets = np.array(targets).take([0, 1, 2, 3, 4])
utl.orthogonal_feature_scans(
    loss_model, scaler_in, scaler_out, features, targets, plot_targets)
plt.savefig(output_dir + 'FeatureScans_individBLMs.pdf')
plt.show()
