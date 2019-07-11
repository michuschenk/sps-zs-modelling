import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from shutil import copy2
import joblib as jbl

from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, Dropout
from keras.optimizers import Adam
from keras.initializers import glorot_normal
from keras.callbacks import EarlyStopping

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle

import utils as utl

import seaborn as sns
sns.set(context='talk', font_scale=0.75)
sns.set_style('white')


# *****************
# (0) Basic config.
# Some network configurations that perform reasonably well
nn_topologies = {
    '1HL_noDO': {'n_nodes_1': 7, 'n_nodes_2': 0, 'dropout': False},
    '2HL_wDO': {'n_nodes_1': 40, 'n_nodes_2': 20, 'dropout': True},
    '2HL_wDO_small': {'n_nodes_1': 20, 'n_nodes_2': 10, 'dropout': True}
}
# training_params = {
#     'early_stopping': False, 'batch_size': 10, 'n_epochs': 1200,   # 1200,
#     'learning_rate': 5e-4
# }
training_params = {
    'early_stopping': True, 'batch_size': 10, 'n_epochs': 600,   # 1200,
    'learning_rate': 8e-4
}

config = nn_topologies['2HL_wDO']
config.update(training_params)

plot_training_data = True
group_duplicates = False

# Place to save code, plots, etc., for reproducibility
# Change output_dir to avoid overwriting
output_dir = 'output/004/'
Path(output_dir).mkdir(parents=True, exist_ok=True)
copy2('main.py', output_dir + 'main.py.state')
copy2('utils.py', output_dir + 'utils.py.state')

features = ['ZS1.LSS2.ANODE:DO_PPM',  # 'ZS1.LSS2.ANODE:UP_PPM',
            'ZS2.LSS2.ANODE:DO_PPM', 'ZS2.LSS2.ANODE:UP_PPM',
            'ZS3.LSS2.ANODE:DO_PPM', 'ZS3.LSS2.ANODE:UP_PPM',
            'ZS4.LSS2.ANODE:DO_PPM', 'ZS4.LSS2.ANODE:UP_PPM',
            'ZS5.LSS2.ANODE:DO_PPM', 'ZS5.LSS2.ANODE:UP_PPM',
            'ZS.LSS2.GIRDER:DO_PPM']

targets = ['SPS.BLM.21636.ZS1:LOSS_CYCLE_NORM',
           'SPS.BLM.21652.ZS2:LOSS_CYCLE_NORM',
           'SPS.BLM.21658.ZS3:LOSS_CYCLE_NORM',
           'SPS.BLM.21674.ZS4:LOSS_CYCLE_NORM',
           'SPS.BLM.21680.ZS5:LOSS_CYCLE_NORM']   #,
           # 'SPS.BLM.21694.TCE:LOSS_CYCLE_NORM',
           # 'SPS.BLM.21772.TPST:LOSS_CYCLE_NORM']


# ********************
# (1) Load TIMBER data
# For now: anode positions, loss for each ZS BLM
# TODO: may need a lot more cleaning / pre-processing
train_data_source = {
    'filepath': './timber_data/',
    'filenames': ['TIMBER_DATA_011118_1300-1600.xls',
                  # 'TIMBER_DATA_270318_1745-280318_0130.xls',
                  'TIMBER_DATA_260418_1730-2300.xls'],
    'start_times': ['2018-11-01 13:18:00.000',
                    # '2018-03-27 19:20:00.000',
                    '2018-04-26 17:30:00.000'],
    'end_times': ['2018-11-01 15:38:00.000',
                  # '2018-03-28 01:30:00.000',
                  '2018-04-26 23:00:00.000']
}

train_data = utl.load_data(train_data_source)
train_data = utl.filter_zs_blm_outliers(train_data, threshold=5e-15)

x_train, y_train = train_data[features], train_data[targets]

if plot_training_data:
    # Plot every dataset (i.e. from different days) separately
    for i in range(len(train_data_source['filenames'])):
        t_start = np.datetime64(train_data_source['start_times'][i])
        t_end = np.datetime64(train_data_source['end_times'][i])
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
y_train = scaler_out.fit_transform(y_train)

# Save scalers to reimport when reusing model
jbl.dump(scaler_in, output_dir + 'scaler_in.save')
jbl.dump(scaler_out, output_dir + 'scaler_out.save')


# **************************************
# (3) Define neural network architecture
# TODO: Try other regressors
n_features = x_train.shape[1]
n_targets = y_train.shape[1]

loss_model = Sequential()

# 1st hidden layer
loss_model.add(Dense(config['n_nodes_1'], input_shape=(n_features,),
                     kernel_initializer=glorot_normal(seed=0)))
loss_model.add(LeakyReLU(alpha=0.2))

# Dropout layer
if config['dropout']:
    loss_model.add(Dropout(rate=0.2, noise_shape=None, seed=None))

# 2nd hidden layer
if config['n_nodes_2']:
    loss_model.add(Dense(config['n_nodes_2'],
                         kernel_initializer=glorot_normal(seed=2)))
    loss_model.add(LeakyReLU(alpha=0.2))

# Output layer
loss_model.add(
    Dense(n_targets, kernel_initializer=glorot_normal(seed=2)))

# Optimiser and loss
adam_opt = Adam(lr=config['learning_rate'], beta_1=0.9, beta_2=0.999,
                epsilon=None, amsgrad=False)
loss_model.compile(optimizer=adam_opt, loss='mean_squared_error',
                   metrics=['acc'])

# Training
callbacks = []
if config['early_stopping']:
    callbacks.append(EarlyStopping(patience=30))

training_history = loss_model.fit(
    x_train, y_train, validation_split=0.15,
    epochs=config['n_epochs'],
    callbacks=callbacks,
    batch_size=config['batch_size'])

# Visualise training evolution
training_history = pd.DataFrame(data=training_history.history)
utl.plot_training_evolution(training_history)
plt.savefig(output_dir + 'training_history.pdf')
plt.show()
loss_model.save(output_dir + 'NN_model')


# ********************************
# (4) Make predictions on test set
# test_data_source = {
#     'filepath': './timber_data/',
#     'filenames': ['TIMBER_DATA_050418_2110-2315.xls'],
#     'start_times': ['2018-04-05 21:10:00.000'],
#     'end_times': ['2018-04-05 23:15:00.000']
# }
# test_data_source = {
#     'filepath': './timber_data/',
#     'filenames': ['TIMBER_DATA_300318_1500-2130.xls'],
#     'start_times': ['2018-03-30 15:00:00.000'],
#     'end_times': ['2018-03-30 21:30:00.000']
# }
# test_data_source = {
#     'filepath': './timber_data/',
#     'filenames': ['TIMBER_DATA_220918_1235-1305_GIRDER.xls'],
#     'start_times': ['2018-09-22 12:35:00.000'],
#     'end_times': ['2018-09-22 13:05:00.000']
# }
test_data_source = {
    'filepath': './timber_data/',
    'filenames': ['TIMBER_DATA_270318_1745-280318_0130.xls'],
    'start_times': ['2018-03-27 19:20:00.000'],
    'end_times': ['2018-03-28 01:30:00.000']
}

test_data = utl.load_data(test_data_source)
test_data = utl.filter_zs_blm_outliers(test_data, threshold=5e-15)
x_test, y_test = test_data[features], test_data[targets]

# ***************
# (4a) Total loss
test_data = utl.add_total_loss(test_data, targets)
t_start = np.datetime64(test_data_source['start_times'][0])
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


# ********************
# (4b) Individual BLMs
# Reload test set
test_data = utl.load_data(test_data_source)
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


"""
# *********************
# (4c) Orthogonal scans
# Perform sanity tests: fake scans for all anodes
utl.orthogonal_feature_scans(loss_model, scaler_in, scaler_out)
plt.savefig(output_dir + 'FeatureScans_individBLMs.pdf')
plt.show()
"""