import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def date_parser(x):
    """ Helper function for date / time parsing of input files. May no
    longer be required. """
    return pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')


def load_data(data_source):
    """ Read data from .xls files into pandas data frame. Time range can
     be adjusted in data_source dict. """
    data = pd.DataFrame()
    filepath = data_source['filepath']
    filenames = data_source['filenames']
    start_times = data_source['start_times']
    end_times = data_source['end_times']
    for i, f in enumerate(filenames):
        data_tmp = pd.read_excel(
            filepath + f, parse_dates=['Timestamp (UTC_TIME)'],
            date_parser=date_parser)

        # Mask times where useful data is
        t_start = np.datetime64(start_times[i])
        t_end = np.datetime64(end_times[i])
        mask_time = ((data_tmp['Timestamp (UTC_TIME)'] >= t_start) &
                     (data_tmp['Timestamp (UTC_TIME)'] <= t_end))
        print('{:s}, using {:d} samples'.format(f, np.sum(mask_time)))
        data_tmp = data_tmp[mask_time]
        data = data.append(data_tmp)
    return data


def separate_features_targets(data):
    """ Define features and targets: remove corresponding columns from
    data frame and return separated features (x) and targets (y). """
    drop_cols_features = list(data.filter(like='LOSS').columns)
    drop_cols_features.append('Timestamp (UTC_TIME)')
    drop_cols_features.append('ZS1.LSS2.ANODE:UP_PPM')

    drop_cols_targets = list(data.filter(like='ANODE').columns)
    drop_cols_targets.append('Timestamp (UTC_TIME)')

    x = data.drop(columns=drop_cols_features)
    y = data.drop(columns=drop_cols_targets)

    return x, y


def orthogonal_feature_scans(loss_model, scaler_in, scaler_out):
    """ Perform fake orthogonal scans for every feature and plot response
    for trained model. Assuming that all features are anode positions
    here. Some hard-coded axis limits , labels, etc. """
    feature_map = {'ZS1_DO': 0, 'ZS2_DO': 1, 'ZS2_UP': 2, 'ZS3_DO': 3,
                   'ZS3_UP': 4, 'ZS4_DO': 5, 'ZS4_UP': 6, 'ZS5_DO': 7,
                   'ZS5_UP': 8}

    axs = plt.subplots(6, 9, figsize=(16, 15), sharex=True)
    fig = axs[0]
    axs = axs[1]

    for j, k in enumerate(feature_map.keys()):
        ind_ZS = feature_map[k]

        n_samples = 50
        n_features = len(feature_map.keys())
        x_test = np.zeros((n_samples, n_features))
        x_test[:, ind_ZS] = np.linspace(-2, 2, n_samples)
        y_pred = loss_model.predict(scaler_in.transform(x_test))
        y_pred = scaler_out.inverse_transform(y_pred)

        for i in range(0, 5):
            ax = axs[i, j]
            ax.plot(x_test[:, ind_ZS], y_pred[:, i], c='dodgerblue')
            if j == 0:
                ax.set_ylabel('ZS{:d} BLM\n(Gy/charge)'.format(i + 1))
                ax.ticklabel_format(
                    style='sci', axis='y', scilimits=(0, 0),
                    useMathText=True)
            if j != 0:
                plt.setp(axs[i, j].get_xticklabels(), visible=False)
                plt.setp(axs[i, j].get_yticklabels(), visible=False)
                axs[i, j].yaxis.get_offset_text().set_visible(False)
            ax.set_ylim(0., 6e-14)
        axs[-1, j].plot(x_test[:, ind_ZS], np.sum(y_pred, axis=1),
                        c='forestgreen')
        axs[-1, j].set_ylim(0., 2e-13)
        if j == 0:
            axs[-1, j].ticklabel_format(
                style='sci', axis='y', scilimits=(0, 0),
                useMathText=True)
        if j == 0:
            axs[-1, j].set_ylabel('Total loss\n(Gy/charge)')
        if j != 0:
            plt.setp(axs[-1, j].get_yticklabels(), visible=False)
            axs[-1, j].yaxis.get_offset_text().set_visible(False)
        axs[-1, j].set_xlabel('{:s} (mm)'.format(k))
        plt.suptitle('Orthogonal feature scans', fontsize=15)

    plt.subplots_adjust(bottom=0.1, left=0.08, top=0.92, right=0.97,
                        hspace=0.23)


def filter_blm_outliers(train_data, threshold=5e-15):
    """ Clean up unreasonable BLM values: when plotting raw data,
    observed some low-loss spikes (~ 1e-15) for all BLMs at same time --
    considered unphysical, hence remove these samples.
    """
    for col in train_data.filter(like='LOSS_CYCLE_NORM'):
        train_data = train_data[train_data[col] > threshold]
    train_data = train_data.reset_index(drop=True)
    return train_data


def add_total_loss(test_data):
    """ Add new column to data frame with total loss of ZS1 ... ZS5
    BLMs. """
    test_data['TOTAL_LOSS_NORM'] = (
        test_data.filter(like='LOSS_CYCLE_NORM').sum(axis=1))
    test_data = test_data.reset_index(drop=True)
    return test_data


def group_duplicates(x_train, y_train, abs_diff=0.001):
    """
    Go through all anode states and group duplicates (discretise with
    given abs_diff value). This can be dangerous operation since not
    all the dynamics is captured in the given model: some states that
    are considered identical may give different BLM output (since
    there are missing features). Best is to analyse BLM signals
    for anode states that are considered identical.
    """
    # TODO: Improve implementation of group_duplicates(..) function

    # When to consider duplicate: atol gives abs delta required for
    # different state.
    n_samples = x_train.shape[0]
    n_duplicates = 0
    unique_states = {}
    unique_targets = {}
    # unique_samples = {}
    for i in range(n_samples):
        current_state = np.array(x_train.iloc[i])
        current_target = np.array(y_train.iloc[i])
        found_duplicate = False
        for k in unique_states.keys():
            state = np.array(unique_states[k])
            target = np.array(unique_targets[k])
            if len(state.shape) > 1:
                state = state[0]
            if np.allclose(current_state, state, atol=abs_diff, rtol=0.):
                found_duplicate = True
                break
        if found_duplicate:
            n_duplicates += 1
            # Overwrite existing state with mean value
            # unique_states[k] = np.mean(
            #     np.array([state, current_state]), axis=0)
            unique_states[k] = np.vstack((unique_states[k],
                                          current_state))
            # Same for the target variable (the BLMs)
            # unique_targets[k] = np.mean(
            #     np.array([target, current_target]), axis=0)
            unique_targets[k] = np.vstack((unique_targets[k],
                                           current_target))
        else:
            key_new = len(unique_states.keys()) + 1
            unique_states[key_new] = current_state
            unique_targets[key_new] = current_target

    print('Total number of samples', n_samples)
    print('Found {:d} duplicates'.format(n_duplicates))

    # Find samples where state is considered 'identical' but target is
    # different ... To check consistency of 'duplicates'
    for k in unique_targets.keys():
        if len(unique_targets[k].shape) > 1:
            n_equals = unique_targets[k].shape[0]
            rel_diff = (np.std(unique_targets[k], axis=0) /
                        np.mean(unique_targets[k], axis=0))
            if np.any(rel_diff > 0.05) and n_equals > 3:
                print('WARNING: Potential data inconsistency at key', k)

    # Take mean
    for k in unique_targets.keys():
        if len(unique_targets[k].shape) > 1:
            unique_targets[k] = np.mean(unique_targets[k], axis=0)
            unique_states[k] = np.mean(unique_states[k], axis=0)

    x_train = pd.DataFrame(
        data=unique_states.values(), columns=x_train.columns)
    y_train = pd.DataFrame(
        data=unique_targets.values(), columns=y_train.columns)

    return x_train, y_train


def plot_data(data, title=''):
    """ Plot all anodes in separate plots alongside total loss. """
    axs = plt.subplots(6, 1, figsize=(9, 15), sharex=True)
    fig = axs[0]
    axs = axs[1].flatten()
    for i, ax in enumerate(axs[:-1]):
        alpha = 1.
        if i == 0:
            alpha = 0.3
        data.plot(x='Timestamp (UTC_TIME)',
                  y='ZS{:d}.LSS2.ANODE:UP_PPM'.format(i+1), ax=ax,
                  c='mediumblue', alpha=alpha,
                  label='UP', legend=None)
        data.plot(x='Timestamp (UTC_TIME)',
                  y='ZS{:d}.LSS2.ANODE:DO_PPM'.format(i+1), ax=ax,
                  c='forestgreen',
                  label='DO', legend=None)

        if i == 1:
            ax.legend(loc='upper left', bbox_to_anchor=(1., 2.3))

        ax.set_ylabel('ZS{:d} anode\npos. (mm)'.format(i+1))
        ax.set_ylim(-2, 2)

    # Total loss
    data['TOTAL_LOSS_NORM'] = (data
                               .filter(like='LOSS_CYCLE_NORM')
                               .sum(axis=1))
    data.plot(x='Timestamp (UTC_TIME)',
              y='TOTAL_LOSS_NORM', ax=axs[-1], c='darkred',
              label='Ground\ntruth', legend=None)
    # axs[-1].set_ylabel('Loss (Gray/charge)')
    axs[-1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0),
                             useMathText=True)
    axs[-1].set_ylabel('Total norm. loss\n(Gy/charge)')
    axs[-1].set_ylim(0, 1.5e-13)
    data.drop(columns=['TOTAL_LOSS_NORM'])
    plt.subplots_adjust(bottom=0.1, left=0.12, top=0.92, right=0.81,
                        hspace=0.23)
    plt.suptitle(title, fontsize=15)

    return fig, axs


def plot_training_evolution(history):
    """ Plot NN loss and accuracy (training and validation) as function
    of epoch. """
    axs = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    fig = axs[0]
    axs = axs[1].flatten()
    history.plot(
        y=['loss', 'val_loss'], ax=axs[0], color=["r", "b"])
    history.plot(
        y=['acc', 'val_acc'], ax=axs[1], color=["r", "b"])
    axs[0].set(ylabel="Loss")
    axs[0].set_yscale("log")
    axs[1].set(xlabel="Epoch", ylabel="Accuracy")
    axs[0].legend(["Training", "Validation"], bbox_to_anchor=(1., 1.05),
                  loc='upper left')

    for ax in axs:
        ax.set_xlim(0, history.shape[0])
        # ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        # ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    axs[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0),
                            useMathText=True)

    plt.setp(axs[0].get_xticklabels(), visible=False)
    plt.subplots_adjust(left=0.14, bottom=0.12, top=0.89, right=0.76)
    plt.suptitle('Training evolution', fontsize=15)


def plot_individual_blms(data, title):
    """ Plot all anodes in same plot alongside individual BLMs. """
    axs = plt.subplots(6, 1, figsize=(9, 15), sharex=True)
    fig = axs[0]
    axs = axs[1].flatten()

    for i in range(1, 6):
        data.plot(x='Timestamp (UTC_TIME)',
                  y='ZS{:d}.LSS2.ANODE:DO_PPM'.format(i),
                  legend=None, ax=axs[0], lw=2, alpha=0.8,
                  label='ZS{:d}: DO'.format(i))
        data.plot(x='Timestamp (UTC_TIME)',
                  y='ZS{:d}.LSS2.ANODE:UP_PPM'.format(i),
                  legend=None, ax=axs[0], lw=2, alpha=0.8,
                  label='ZS{:d}: UP'.format(i))
    axs[0].legend(bbox_to_anchor=(1., 1.05), loc='upper left')

    for i, ax in enumerate(axs[1:]):
        col_to_plot = (
            data.filter(like='ZS{:d}:LOSS_CYCLE_NORM'.format(i + 1))
                .columns[0])
        data.plot(x='Timestamp (UTC_TIME)', y=col_to_plot, ax=ax, lw=2,
                  c='darkred', label='Ground truth', legend=None)

        ax.set_ylabel('ZS{:d} BLM\n(Gy/charge)'.format(i + 1))
        ax.set_ylim(0, 4e-14)
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0),
                            useMathText=True)
    plt.subplots_adjust(bottom=0.1, left=0.12, top=0.92, right=0.79,
                        hspace=0.23)
    plt.suptitle(title, fontsize=15)

    return fig, axs
