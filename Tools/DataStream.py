# !/usr/bin/python3.6
# -*- coding: UTF-8 -*-
import numpy as np
import time
import os
import pandas as pd
import warnings
import gc
import sys
path = r'/home/alex/桌面/Python/'
sys.path.append(path)


class DataStream(object):
    __slots__ = 'instances', 'dict_instances', 'options', 'isShuffle', 'isLoop', 'isSort', 'num_instances', \
                'num_batch', 'vector_dim', 'cur_pointer', 'index_array', 'batches', 'columns', 'num_thres', \
                'float_format'

    def __init__(self, isShuffle=True, isLoop=False, isSort=False, options=None):
        if not options: raise ValueError('Options cannot be empty!')
        # attribute
        self.instances = None
        self.dict_instances = None
        self.options = options
        if options.float_format == 'float32':
            self.float_format = np.float32
        else:
            self.float_format = np.float64
        self.isShuffle = isShuffle
        self.isLoop = isLoop
        self.isSort = isSort

        self.num_instances = None
        self.num_batch = None
        self.vector_dim = 0
        self.cur_pointer = None
        self.index_array = []
        self.batches = []

        self.columns = None
        self.num_thres = 0

    def read_instances(self, path
                       , header: bool = True
                       , sep=','
                       , mode: str = 'new'
                       , method: str = 'feed'
                       , sampling: str = 'Down'
                       , set_type: str = 'train'
                       , num_threshold=None):
        """
        mode:
            new: initiate a new object.
            add: add new data into the existing object.
        sampling:
            'Up': up-sampling
            'Down': down-sampling
        method: feed: data info saved as a list
                purge: data info saved as a dictionary
        """
        # para
        time_start = time.time()
        if header:
            header_new = 0
        else:
            header_new = None

        if method == 'feed':
            self.read_instances_feed(path, header=header_new, sep=sep, mode=mode, set_type=set_type,
                                     num_thres=num_threshold)

        elif method == 'purge':
            assert self.options.is_classifier, 'Purge method is not necessary for non-classification problems!'
            self.read_instances_purge(path, header=header_new, sep=sep, mode=mode, set_type=set_type,
                                      sampling=sampling, num_thres=num_threshold)

        else:
            raise ValueError("Unsupported method: %s" % method)
        time_end = time.time()
        print('Reading file: %s complete! Time consume: %.2f seconds' % (os.path.split(path)[1], (time_end - time_start)))

    def read_instances_feed(self, path, header, sep, mode, set_type, num_thres):
        """
        Format: Time Ticker Var... Label
        Sep : file separated by sep
        Mode: 'add' or 'new' add data to existed data or build new dataset
        """
        # initialized
        assert self.options.data_type == 'csv' or self.options.data_type == 'mmap', \
            f'Unsupported data type: {self.options.data_type}!'
        if mode == 'new':
            if self.instances:
                del self.instances
            gc.collect()

            self.instances = None

        if self.options.data_type == 'csv':
            data_set = pd.read_csv(path, dtype={0: str, 1: str}, sep=sep, header=header)
            if header == 0:
                header_column = data_set.columns.tolist()
            else:
                header_column = list(range(len(data_set.columns)))
            data_set = data_set.values

        else:
            assert set_type == 'train' or set_type == 'dev', f'Unsupported mode: {set_type}!'
            if set_type == 'train':
                data_set = np.array(np.memmap(path, dtype=np.float64, mode='r', shape=(
                    self.options.train_row_count, self.options.column_count+5)))
            else:
                data_set = np.array(np.memmap(path, dtype=np.float64, mode='r', shape=(
                    self.options.dev_row_count, self.options.column_count+5)))
            if header == 0:
                warnings.warn('Memory-map format may has no header, check again the value of parameter header!')
                header_column = data_set[0].tolist()
                num = np.where(np.isnan(data_set[1:, 2]))[0][0]
                data_set = np.array(data_set[1:num])
            else:
                header_column = list(range(len(data_set[0].tolist())))
                num = np.where(np.isnan(data_set[0:, 2]))[0][0]
                data_set = np.array(data_set[0:num])

        if mode != 'new':
            assert self.columns == header_column, '加入列名需与原来的数据相同!'
        else:
            self.columns = header_column

        if num_thres:
            instances = data_set[0:num_thres]
        else:
            instances = data_set

        del data_set
        gc.collect()

        if mode == 'new':
            self.instances = instances
        else:
            self.instances = np.vstack((self.instances, instances))

        del instances
        gc.collect()

    def read_instances_purge(self, path, header, sep, mode, sampling, set_type, num_thres):
        """
        Format: Time Ticker Var... Label
        Sep : file separated by sep
        Mode: 'add' or 'new' add data to existed data or build new dataset
        Method: 'Up' means up-sampling and 'Down' means down-sampling
        """

        # initialized
        assert self.options.data_type == 'csv' or self.options.data_type == 'mmap', \
            f'Unsupported data type: {self.options.data_type}!'
        assert mode == 'add' or mode == 'new', f'Unsupported mode: {mode}!'
        assert sampling == 'Up' or sampling == 'Down', f'Unsupported sampling method: {sampling}!'

        if mode == 'new':
            if self.instances:
                del self.instances
            if self.dict_instances:
                del self.dict_instances
            gc.collect()

            self.instances = None
            self.dict_instances = dict()
        else:
            self.instances = None

        if self.options.data_type == 'csv':
            data_set = pd.read_csv(path, dtype={0: str, 1: str}, sep=sep, header=header)
            if header == 0:
                header_column = data_set.columns.tolist()
            else:
                header_column = list(range(len(data_set.columns)))

        else:
            assert set_type == 'train' or set_type == 'dev', f'Unsupported mode: {set_type}!'
            if set_type == 'train':
                data_set = np.array(np.memmap(path, dtype='float64', mode='r', shape=(
                    self.options.train_row_count, self.options.column_count+5)))
            else:
                data_set = np.array(np.memmap(path, dtype='float64', mode='r', shape=(
                    self.options.dev_row_count, self.options.column_count+5)))
            if header == 0:
                warnings.warn('Memory-map format may has no header, check again the value of parameter header!')
                header_column = data_set[0].tolist()
                num = np.where(np.isnan(data_set[1:, 2]))[0][0]
                data_set = np.array(data_set[1:num])
            else:
                header_column = list(range(len(data_set[0].tolist())))
                num = np.where(np.isnan(data_set[1:, 2]))[0][0]
                data_set = np.array(data_set[0:num])

            data_time = np.array(list(map(
                lambda x, y: str(time.strftime('%Y-%m-%d %H:%M:%S',
                                               time.strptime(str(int(x/10**3)).zfill(8) + str(int(y/10**3)).zfill(6),
                                                             '%Y%m%d%H%M%S'))),
                data_set[:, 0], data_set[:, 1])))
            data_ticker = np.array(list(map(lambda x: str(int(x/10**3)).zfill(6), data_set[:, 2])))
            data_set = data_set[:, 1:].astype(object)
            data_set[:, 0] = data_time
            data_set[:, 1] = data_ticker
            data_set = pd.DataFrame(data_set, columns=header_column)

        if mode != 'new':
            assert self.columns == header_column, '加入列名需与原来的数据相同!'
        else:
            self.columns = header_column

        # make data
        key_target = self.columns[-(self.options.num_fit_target + 1)]
        grouped = data_set.groupby(key_target)

        # handle data
        len_list = {key: group.shape[0] for key, group in grouped}
        if num_thres:
            num_thres_grouped = num_thres // len(len_list.values())
            if sampling == 'Up':
                num = min(max(len_list.values()), num_thres_grouped)
            else:
                num = min(min(len_list.values()), num_thres_grouped)
        else:
            if sampling == 'Up':
                num = max(len_list.values())
            else:
                num = min(len_list.values())

        for key, group in grouped:
            if key in self.dict_instances.keys():
                self.dict_instances[key] = np.concatenate((self.dict_instances[key], group.values[0:num]))
            else:
                self.dict_instances[key] = group.values[0:num]

        for key in self.dict_instances.keys():
            if num <= len_list[key]:
                replace_flag = False
            else:
                replace_flag = True
            indexes = np.random.choice(self.dict_instances[key].shape[0], size=self.num_thres + num,
                                       replace=replace_flag).tolist()
            if self.instances is not None:
                self.instances = np.concatenate((self.instances, self.dict_instances[key][indexes, :]))
            else:
                self.instances = self.dict_instances[key][indexes, :]

        self.num_thres += num
        del data_set, grouped
        gc.collect()

    def re_sampling(self, sampling='Down', num_thres=1000000):
        if not self.dict_instances:
            raise ValueError("Dict_instances don't exist!")

        self.instances = []
        len_list = {item: self.dict_instances[item].shape[0] for item in self.dict_instances}
        if num_thres:
            num_thres_grouped = num_thres // len(len_list.items())
            if sampling == 'Up':
                num = min(max(len_list), num_thres_grouped)
            else:
                num = min(min(len_list), num_thres_grouped)
        else:
            if sampling == 'Up':
                num = max(len_list)
            else:
                num = min(len_list)

        for key in self.dict_instances.keys():
            if num <= len_list[key]:
                replace_flag = False
            else:
                replace_flag = True
            indexes = np.random.choice(self.dict_instances[key].shape[0], size=num, replace=replace_flag).tolist()
            if self.instances:
                self.instances = np.concatenate((self.instances, self.dict_instances[key][indexes, :]))
            else:
                self.instances = self.dict_instances[key][indexes, :]

    def load_data(self, data):
        self.instances = data

    def make_batches(self, trading=False):
        del self.dict_instances
        self.dict_instances = None
        gc.collect()

        # sort instances based on sentence length
        if self.isSort:  # sort instances based on ticker and Timestamp
            self.instances = sorted(self.instances, key=lambda instance: (self.instances[1], self.instances[0]))
        else:
            if self.isShuffle: np.random.shuffle(self.instances)

        # distribute into different buckets
        self.num_instances = self.instances.shape[0]
        batch_spans = DataStream.split_batches(self.num_instances, self.options.batch_size)
        self.batches = []
        cur_batch = None
        for batch_index, (batch_start, batch_end) in enumerate(batch_spans):
            cur_instances = self.instances[batch_start:batch_end]
            if trading is False:
                cur_batch = InstanceBatch(cur_instances, self.options)
            else:
                cur_batch = InstanceBatch_trading(cur_instances, self.options)
            cur_instances = None
            self.batches.append(cur_batch)

        self.vector_dim = len(cur_batch.vector[0])
        self.num_batch = len(self.batches)
        self.index_array = np.arange(self.num_batch)
        self.cur_pointer = 0

        del self.instances
        self.instances = None
        gc.collect()

    def nextBatch(self):
        if self.cur_pointer >= self.num_batch:
            if not self.isLoop: return None
            self.cur_pointer = 0
        cur_batch = self.batches[self.index_array[self.cur_pointer]]
        self.cur_pointer += 1
        return cur_batch

    def reset(self):
        self.cur_pointer = 0

    def shuffle(self):
        np.random.shuffle(self.index_array)
        self.cur_pointer = 0

    def get_num_batch(self):
        return self.num_batch

    def get_num_instance(self):
        try:
            return self.instances.shape[0]
        except AttributeError:
            return self.num_instances

    def get_batch(self, i):
        if i >= self.num_batch: return None
        return self.batches[self.index_array[i]]

    def get_vector_dim(self):
        return self.vector_dim

    def to_csv(self, output_path, verbose=True):
        if self.instances.size == 0: warnings.warn('Instances set is empty!')
        folder, filename = os.path.split(output_path)
        if not os.path.exists(folder): os.makedirs(folder)
        if verbose:
            print('数据输出路径为：%s' % output_path)
        pd.DataFrame(self.instances, columns=self.columns).to_csv(
            output_path, sep=',', mode='w', header=True, index=False, encoding='utf-8')

    def to_parquet(self, output_path, verbose=True):
        if self.instances.size == 0: warnings.warn('Instances set is empty!')
        folder, filename = os.path.split(output_path)
        if not os.path.exists(folder): os.makedirs(folder)
        if verbose:
            print('数据输出路径为：%s' % output_path)
        pd.DataFrame(self.instances, columns=self.columns).to_parquet(output_path)

    def to_mmap(self, output_path, mode: str = 'train', verbose=True):
        assert mode == 'train' or mode == 'dev', f'Unsupported mode: {mode}!'
        if self.instances.size == 0: warnings.warn('Instances set is empty!')
        folder, filename = os.path.split(output_path)
        if not os.path.exists(folder): os.makedirs(folder)
        if verbose: print('数据输出路径为：%s' % output_path)
        if mode == 'train':
            data_shape = (self.options.train_row_count, self.options.column_count+5)
        else:
            data_shape = (self.options.dev_row_count, self.options.column_count+5)

        data_temp = np.full(data_shape, np.nan, dtype=np.float64)
        data_temp[0:self.instances.shape[0], 3:] = self.instances[:, 2:]
        date_temp = np.array(
            list(map(lambda x: (int(time.strftime("%Y%m%d", time.strptime(x, "%Y-%m-%d %H:%M:%S")))*1000,
                                int(time.strftime("%H%M%S", time.strptime(x, "%Y-%m-%d %H:%M:%S")))*1000),
                     self.instances[:, 0]))).astype(np.float64)
        data_temp[0:self.instances.shape[0], [0, 1]] = date_temp
        data_temp[0:self.instances.shape[0], 2] = \
            np.array(list(map(lambda x: int(x)*1000, self.instances[:, 1]))).astype(np.float64)
        dat = np.memmap(output_path, dtype=np.float64, mode='w+', shape=data_shape)
        dat[:] = data_temp

    def check_status(self, sampling=True):
        print('=' * 30)
        print('Data stream status')
        print('=' * 30)
        print('Number of instances：%6s' % str(self.get_num_instance()))
        print('Number of batches：%3s' % str(self.get_num_batch()))
        print('Current pointer：%3s' % str(self.cur_pointer))
        print('Number of vector dimensions：%4s' % str(self.get_vector_dim()))
        if self.options.is_fit is False:
            print('数据流应用类型： 分类')
        else:
            print('数据流应用类型： 拟合')
        if self.cur_pointer is not None and sampling is True:
            print('=' * 30)
            print('数据流样本输入向量:')
            print(self.get_batch(self.cur_pointer).vector[-1])
            print('数据流样本输入标签:')
            print(self.get_batch(self.cur_pointer).label_truth[-1])
        print('')

    @staticmethod
    def split_batches(size, batch_size):
        nb_batch = int(np.ceil(size / float(batch_size)))
        return [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, nb_batch)]


class InstanceBatch(object):
    __slots__ = 'options', 'datetime', 'ticker', 'batch_size', 'label_truth_0', 'label_truth_1', 'vector'

    def __init__(self, instances, options):
        self.options = options

        if self.options.data_type == 'mmap':
            self.datetime = np.array(list(map(
                lambda x, y: str(time.strftime('%Y-%m-%d %H:%M:%S',
                                               time.strptime(
                                                   str(int(x/1000)).zfill(8) + str(int(y/1000)).zfill(6),
                                                   '%Y%m%d%H%M%S'))),
                instances[:, 0], instances[:, 1]))).copy()
            self.ticker = np.array(list(map(lambda x: str(int(x/1000)).zfill(6), instances[:, 2]))).copy()
            instances = instances[:, 3:].astype(self.options.float_format)

            self.batch_size = instances.shape[0]

            if self.options.is_classifier and not self.options.is_fit:
                self.label_truth_0 = instances[:, -1]
                self.label_truth_1 = None
                self.vector = instances[:, 0:-1]
            elif self.options.is_fit and self.options.is_classifier:
                self.label_truth_0 = instances[:, -(1 + self.options.num_fit_target)]
                self.label_truth_1 = instances[:, -self.options.num_fit_target:]
                self.vector = instances[:, 0:-(1 + self.options.num_fit_target)]
            elif self.options.is_fit and not self.options.is_classifier:
                self.label_truth_0 = None
                self.label_truth_1 = instances[:, -self.options.num_fit_target:]
                self.vector = instances[:, 0:-self.options.num_fit_target]
            else:
                raise ValueError(
                    'Option.is_fit and option.is_classifier cannot be False at the same time, check option file!')
        else:
            self.datetime = instances[:, 0]
            self.ticker = instances[:, 1]
            self.batch_size = instances.shape[0]
            if self.options.is_classifier and not self.options.is_fit:
                self.label_truth_0 = instances[:, -1]
                self.label_truth_1 = None
                self.vector = instances[:, 2:-1]
            elif self.options.is_fit and self.options.is_classifier:
                self.label_truth_0 = instances[:, -(1 + self.options.num_fit_target)]
                self.label_truth_1 = instances[:, -self.options.num_fit_target:]
                self.vector = instances[:, 2:-(1 + self.options.num_fit_target)]
            elif self.options.is_fit and not self.options.is_classifier:
                self.label_truth_0 = None
                self.label_truth_1 = instances[:, -self.options.num_fit_target:]
                self.vector = instances[:, 2:-self.options.num_fit_target]
            else:
                raise ValueError(
                    'Option.is_fit and option.is_classifier cannot be False at the same time, check option file!')
            self.vector = self.vector.astype(self.options.float_format)


class InstanceBatch_trading(object):
    def __init__(self, instances, options):
        self.options = options
        self.ticker = instances[:, 0]
        self.batch_size = instances.shape[0]

        if self.options.is_classifier and not self.options.is_fit:
            self.label_truth_0 = np.zeros(instances.shape[0])
            self.label_truth_1 = None
            self.vector = instances[:, 1:]
        elif self.options.is_fit and self.options.is_classifier:
            self.label_truth_0 = np.zeros(instances.shape[0])
            self.label_truth_1 = np.zeros((instances.shape[0], self.options.num_fit_target))
            self.vector = instances[:, 1:]
        elif self.options.is_fit and not self.options.is_classifier:
            self.label_truth_0 = None
            self.label_truth_1 = np.zeros((instances.shape[0], self.options.num_fit_target))
            self.vector = instances[:, 1:]
        else:
            raise ValueError(
                'Option.is_fit and option.is_classifier cannot be False at the same time, check option file!')
        self.vector = self.vector.astype(self.options.float_format)


def test_csv():
    import Project.Tools.namespace_utils as namespace_utils
    config_path = r'/home/alex/桌面/Python/Project/configs/load_csv.config'
    print('Loading the configuration from ' + config_path)
    FLAGS = namespace_utils.load_namespace(config_path)
    path = r'/data/Quant_research/feature_data/normalized_data/dataset_20140201_20140630.csv'
    trainDataStream = DataStream(isShuffle=True, isLoop=False, isSort=False, options=FLAGS)
    print('Build Training DataStream ... | %s' % path)
    trainDataStream.read_instances(path, header=True, sep=',', mode='new', method='feed')
    trainDataStream.read_instances(path, header=True, sep=',', mode='add', method='feed')
    # trainDataStream.read_instances(path, header=True, sep=',', mode='new', method='purge')
    # trainDataStream.read_instances(path, header=True, sep=',', mode='add', method='purge')
    trainDataStream.make_batches()
    print('Number of instances in trainDataStream: {}'.format(trainDataStream.get_num_instance()))
    print('Number of batches in trainDataStream: {}'.format(trainDataStream.get_num_batch()))
    print('Number of features in trainDataStream: {}'.format(trainDataStream.vector_dim))
    print()


def test_mmap():
    import Project.Tools.namespace_utils as namespace_utils
    config_path = r'/home/alex/桌面/Python/Project/configs/load_mmap.config'
    print('Loading the configuration from ' + config_path)
    FLAGS = namespace_utils.load_namespace(config_path)
    path = r'/data/Quant_research/feature_data/normalized_data/dataset_20140201_20140630.mm'
    trainDataStream = DataStream(isShuffle=True, isLoop=False, isSort=False, options=FLAGS)
    print('Build Training DataStream ... | %s' % path)
    trainDataStream.read_instances(path, header=False, sep=',', mode='new', method='feed')
    trainDataStream.make_batches()
    print('Number of instances in trainDataStream: {}'.format(trainDataStream.get_num_instance()))
    print('Number of batches in trainDataStream: {}'.format(trainDataStream.get_num_batch()))
    print('Number of features in trainDataStream: {}'.format(trainDataStream.vector_dim))
    print()


if __name__ == '__main__':
    test_mmap()
