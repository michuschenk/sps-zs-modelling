import pandas as pd
import numpy as np


class Dataconfig:
    def __init__(self, config):
        self.config = config

    def __add__(self, other):
        new_config = {}
        for k in self.config.keys():
            new_config[k] = self.config[k] + other.config[k]
        return Dataconfig(config=new_config)

    def __radd__(self, other):
        return self.__add__(other)


def date_parser(x):
    """ Helper function for date / time parsing of input files. May no
    longer be required. """
    return pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')


def load_data(dataconfig):
    """ Read data from .xls files into pandas data frame. Time range can
     be adjusted in data_source dict. """
    data = pd.DataFrame()
    filepaths = dataconfig.config['filepaths']
    filenames = dataconfig.config['filenames']
    start_times = dataconfig.config['start_times']
    end_times = dataconfig.config['end_times']
    for i, f in enumerate(filenames):
        data_tmp = pd.read_excel(
            filepaths[i] + f, parse_dates=['Timestamp (UTC_TIME)'],
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


config_dict = {
    '270318': {
        'filepaths': ['./timber_data/'],
        'filenames': ['TIMBER_DATA_270318_1745-280318_0130.xls'],
        'start_times': ['2018-03-27 19:20:00.000'],
        'end_times': ['2018-03-28 01:30:00.000'],
        'comment': ['Best manual anode scan set']
    },

    '300318': {
        'filepaths': ['./timber_data/'],
        'filenames': ['TIMBER_DATA_300318_1500-2130.xls'],
        'start_times': ['2018-03-30 15:00:00.000'],
        'end_times': ['2018-03-30 21:30:00.000'],
        'comment': ['...']
    },

    '050418': {
        'filepaths': ['./timber_data/'],
        'filenames': ['TIMBER_DATA_050418_2110-2315.xls'],
        'start_times': ['2018-04-05 21:10:00.000'],
        'end_times': ['2018-04-05 23:15:00.000'],
        'comment': ['extensive girder DO scans']
    },

    '260418': {
        'filepaths': ['./timber_data/'],
        'filenames': ['TIMBER_DATA_260418_1730-2300.xls'],
        'start_times': ['2018-04-26 17:30:00.000'],
        'end_times': ['2018-04-26 23:00:00.000'],
        'comment': ['general purpose training set, with a bit of girder DO']
    },

    '220918': {
        'filepaths': ['./timber_data/'],
        'filenames': ['TIMBER_DATA_220918_1235-1305_GIRDER.xls'],
        'start_times': ['2018-09-22 12:35:00.000'],
        'end_times': ['2018-09-22 13:05:00.000'],
        'comment': ['some girder movement']
    },

    '011118': {
        'filepaths': ['./timber_data/'],
        'filenames': ['TIMBER_DATA_011118_1300-1600.xls'],
        'start_times': ['2018-11-01 13:18:00.000'],
        'end_times': ['2018-11-01 15:38:00.000'],
        'comment': ['Powell training: multi-param. scans, essential']
    }
}
