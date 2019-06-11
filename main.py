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
    '2HL_wDO': {'n_nodes_1': 40, 'n_nodes_2': 20, 'dropout': True}
}
training_params = {
    'early_stopping': False, 'batch_size': 10, 'n_epochs': 1200,
    'learning_rate': 5e-4
}

config = nn_topologies['1HL_noDO']
config.update(training_params)

plot_training_data = True
group_duplicates = False

# Place to save code, plots, etc., for reproducibility
# Change output_dir to avoid overwriting
output_dir = 'output/002/'
Path(output_dir).mkdir(parents=True, exist_ok=True)
copy2('main.py', output_dir + 'main.py.state')
copy2('utils.py', output_dir + 'utils.py.state')


# ********************
# (1) Load TIMBER data
# For now: anode positions, loss for each ZS BLM
# TODO: may need a lot more cleaning / pre-processing
train_data_source = {
    'filepath': './timber_data/',
    'filenames': ['TIMBER_DATA_011118_1300-1600.xls'],
    'start_times': ['2018-11-01 13:18:00.000'],
    'end_times': ['2018-11-01 15:38:00.000']
}

train_data = utl.load_data(train_data_source)
# train_data = utl.filter_blm_outliers(train_data, threshold=0e-15)

if plot_training_data:
    # Plot every dataset (i.e. from different days) separately
    for i in range(len(train_data_source['filenames'])):
        t_start = np.datetime64(train_data_source['start_times'][i])
        t_end = np.datetime64(train_data_source['end_times'][i])
        mask_time = ((train_data['Timestamp (UTC_TIME)'] >= t_start) &
                     (train_data['Timestamp (UTC_TIME)'] <= t_end))
        sub_data = train_data[mask_time]
        utl.plot_data(data=sub_data, title='Training data\n' +
                                           str(t_start).split('T')[0])
        plt.savefig(output_dir + 'Training_data_set{:d}.pdf'.format(i))

        utl.plot_individual_blms(
            data=sub_data, title='Training data (indiv. BLMs)\n' +
                                 str(t_start).split('T')[0])
        plt.savefig(output_dir + 'Training_data_set{:d}_individBLMs.pdf'
                    .format(i))
        plt.show()


# *****************************
# (2) Prepare data for training
# TODO: Use sklearn pipelines
x_train, y_train, features, targets = (
    utl.separate_features_targets(data=train_data))

if group_duplicates:
    x_train, y_train = utl.group_duplicates(x_train, y_train,
                                            abs_diff=0.001)

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
test_data_source = {
    'filepath': './timber_data/',
    'filenames': ['TIMBER_DATA_270318_1745-280318_0130.xls'],
    'start_times': ['2018-03-27 19:00:00.000'],
    'end_times': ['2018-03-28 01:30:00.000']
}
test_data = utl.load_data(test_data_source)
# test_data = utl.filter_blm_outliers(test_data, threshold=0e-15)

x_test, y_test, features, targets = (
    utl.separate_features_targets(data=test_data))
test_data = utl.add_total_loss(test_data, targets)


# ***************
# (4a) Total loss
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
axs[-1].set_ylim(0.5e-13, 1.4e-13)
axs[-1].legend(loc='upper left', bbox_to_anchor=(1., 1.05))
plt.savefig(output_dir + 'TestData_withPrediction.pdf')
plt.show()

# ********************
# (4b) Individual BLMs
# Reload test set
test_data = utl.load_data(test_data_source)
test_data = utl.filter_blm_outliers(test_data, threshold=5e-15)
test_data = utl.add_total_loss(test_data)

_, axs = utl.plot_individual_blms(
    data=test_data, title='Test data (indiv. BLMs)\n' +
                          str(t_start).split('T')[0])

# Add predictions on top
x_test, y_test = utl.separate_features_targets(data=test_data)
x_test = scaler_in.transform(x_test)
pred_test = loss_model.predict(x_test)
pred_test = scaler_out.inverse_transform(pred_test)

for i, ax in enumerate(axs[1:]):
    ax.plot(test_data['Timestamp (UTC_TIME)'], pred_test[:, i],
            c='dodgerblue', label='Model\nprediction')
    if i == 1:
        ax.legend(loc='upper left', bbox_to_anchor=(1., 1.05))
plt.savefig(output_dir + 'TestData_withPrediction_individBLMs.pdf')
plt.show()

# *********************
# (4c) Orthogonal scans
# Perform sanity tests: fake scans for all anodes
utl.orthogonal_feature_scans(loss_model, scaler_in, scaler_out)
plt.savefig(output_dir + 'FeatureScans_individBLMs.pdf')
plt.show()

