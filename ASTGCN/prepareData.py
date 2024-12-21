import os
import numpy as np
import argparse
import configparser


def search_data(sequence_length, num_of_depend, label_start_idx,
                num_for_predict, units, points_per_hour):
    """
    根据给定的参数搜索数据序列中的特定部分。该函数主要用于根据预测目标的起始索引和需要预测的点数，
    以及历史数据的长度和依赖关系的数量，找出用于预测的历史数据片段。

    Parameters
    ----------
    sequence_length: int, 历史数据的总长度。
    num_of_depend: int, 依赖关系的数量，即需要查找的历史数据片段的数量。
    label_start_idx: int, 预测目标的起始索引。
    num_for_predict: int, 每个样本将预测的点数。
    units: int, 时间单位，例如周: 7 * 24, 天: 24, 近期(小时): 1。
    points_per_hour: int, 每小时的点数，取决于数据的粒度。

    Returns
    ----------
    list[(start_idx, end_idx)]: 一个包含元组的列表，每个元组表示一个历史数据片段的起始和结束索引。
    """

    # 检查每小时的点数是否合法
    if points_per_hour < 0:
        raise ValueError("points_per_hour should be greater than 0!")

    # 检查预测目标的索引加上预测的点数是否超出历史数据的长度
    if label_start_idx + num_for_predict > sequence_length:
        return None

    x_idx = []
    # 遍历每个依赖关系，计算历史数据片段的起始和结束索引
    for i in range(1, num_of_depend + 1):
        start_idx = label_start_idx - points_per_hour * units * i
        end_idx = start_idx + num_for_predict
        # 检查起始索引是否非负
        if start_idx >= 0:
            x_idx.append((start_idx, end_idx))
        else:
            # 如果起始索引小于0，说明无法找到足够的历史数据，返回None
            return None

    # 检查找到的历史数据片段数量是否与依赖关系的数量相等
    if len(x_idx) != num_of_depend:
        return None

    # 返回反转后的历史数据片段列表，以符合时间顺序
    return x_idx[::-1]


def get_sample_indices(data_sequence, num_of_weeks, num_of_days, num_of_hours,
                       label_start_idx, num_for_predict, points_per_hour=12):
    """
    根据给定的数据序列和参数，获取样本索引。

    Parameters
    ----------
    data_sequence: np.ndarray
                   数据序列，形状为 (序列长度, 节点数, 特征数)。
    num_of_weeks, num_of_days, num_of_hours: int
                   分别表示用于样本的周数、天数和小时数。
    label_start_idx: int, 预测目标开始的索引，预测值开始的那个点。
    num_for_predict: int,
                     每个样本将预测的点数。
    points_per_hour: int, default 12, 每小时的点数。

    Returns
    ----------
    week_sample: np.ndarray
                 周样本，形状为 (num_of_weeks * points_per_hour, 节点数, 特征数)。
    day_sample: np.ndarray
                 天样本，形状为 (num_of_days * points_per_hour, 节点数, 特征数)。
    hour_sample: np.ndarray
                 小时样本，形状为 (num_of_hours * points_per_hour, 节点数, 特征数)。
    target: np.ndarray
            目标值，形状为 (num_for_predict, 节点数, 特征数)。
    """
    # 初始化样本变量
    week_sample, day_sample, hour_sample = None, None, None

    # 检查预测目标索引是否超出数据序列范围
    if label_start_idx + num_for_predict > data_sequence.shape[0]:
        return week_sample, day_sample, hour_sample, None

    # 如果周数大于0，获取周样本
    if num_of_weeks > 0:
        # 计算周样本的索引
        week_indices = search_data(data_sequence.shape[0], num_of_weeks,
                                   label_start_idx, num_for_predict,
                                   7 * 24, points_per_hour)
        # 如果周样本索引为空，返回None
        if not week_indices:
            return None, None, None, None

        # 拼接周样本
        week_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in week_indices], axis=0)

    # 如果天数大于0，获取天样本
    if num_of_days > 0:
        # 计算天样本的索引
        day_indices = search_data(data_sequence.shape[0], num_of_days,
                                  label_start_idx, num_for_predict,
                                  24, points_per_hour)
        # 如果天样本索引为空，返回None
        if not day_indices:
            return None, None, None, None

        # 拼接天样本
        day_sample = np.concatenate([data_sequence[i: j]
                                     for i, j in day_indices], axis=0)

    # 如果小时数大于0，获取小时样本
    if num_of_hours > 0:
        # 计算小时样本的索引
        hour_indices = search_data(data_sequence.shape[0], num_of_hours,
                                   label_start_idx, num_for_predict,
                                   1, points_per_hour)
        # 如果小时样本索引为空，返回None
        if not hour_indices:
            return None, None, None, None

        # 拼接小时样本
        hour_sample = np.concatenate([data_sequence[i: j]
                                      for i, j in hour_indices], axis=0)

    # 获取目标值
    target = data_sequence[label_start_idx: label_start_idx + num_for_predict]

    # 返回样本和目标值
    return week_sample, day_sample, hour_sample, target



def read_and_generate_dataset(graph_signal_matrix_filename,
                                                     num_of_weeks, num_of_days,
                                                     num_of_hours, num_for_predict,
                                                     points_per_hour=12, save=False):
    """
    读取并生成数据集函数，用于处理图形信号矩阵数据，生成用于预测模型训练的数据集。

    Parameters
    ----------
    graph_signal_matrix_filename: str, 图形信号矩阵文件的路径
    num_of_weeks, num_of_days, num_of_hours: int, 用于预测模型的周、天、小时数
    num_for_predict: int, 预测的目标维度
    points_per_hour: int, default 12, 每小时的数据点数，取决于数据
    save: bool, 是否保存生成的数据集

    Returns
    ----------
    feature: np.ndarray,
             形状为 (num_of_samples, num_of_depend * points_per_hour,
                       num_of_vertices, num_of_features)
    target: np.ndarray,
            形状为 (num_of_samples, num_of_vertices, num_for_predict)
    """
    # 加载图形信号矩阵数据
    data_seq = np.load(graph_signal_matrix_filename)['data']

    all_samples = []
    for idx in range(data_seq.shape[0]):    # 0~16991
        # 获取样本索引
        sample = get_sample_indices(data_seq, num_of_weeks, num_of_days,
                                    num_of_hours, idx, num_for_predict,
                                    points_per_hour)
        if ((sample[0] is None) and (sample[1] is None) and (sample[2] is None)):
            continue

        week_sample, day_sample, hour_sample, target = sample

        sample = []

        # 处理周样本
        if num_of_weeks > 0:
            week_sample = np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(week_sample)

        # 处理天样本
        if num_of_days > 0:
            day_sample = np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(day_sample)

        # 处理小时样本
        if num_of_hours > 0:
            hour_sample = np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1))  # (1,N,F,T)
            sample.append(hour_sample)

        # 处理目标数据
        target = np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]  # (1,N,T)
        sample.append(target)

        # 处理时间样本
        time_sample = np.expand_dims(np.array([idx]), axis=0)  # (1,1)
        sample.append(time_sample)

        all_samples.append(sample)
        # sampe：[(week_sample),(day_sample),(hour_sample),target,time_sample] = [(1,N,F,Tw),(1,N,F,Td),(1,N,F,Th),(1,N,Tpre),(1,1)]

    print(len(all_samples))
    # 训练集、验证集和测试集的分割点
    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)

    training_set = [np.concatenate(i, axis=0)
                    for i in zip(*all_samples[:split_line1])]  # [(B,N,F,Tw),(B,N,F,Td),(B,N,F,Th),(B,N,Tpre),(B,1)]
    validation_set = [np.concatenate(i, axis=0)
                      for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[split_line2:])]

    # 构建输入数据
    train_x = np.concatenate(training_set[:-2], axis=-1)  # (B,N,F,T')
    val_x = np.concatenate(validation_set[:-2], axis=-1)
    test_x = np.concatenate(testing_set[:-2], axis=-1)

    # 构建目标数据
    train_target = training_set[-2]  # (B,N,T)
    val_target = validation_set[-2]
    test_target = testing_set[-2]

    # 构建时间戳数据
    train_timestamp = training_set[-1]  # (B,1)
    val_timestamp = validation_set[-1]
    test_timestamp = testing_set[-1]

    # 数据归一化
    (stats, train_x_norm, val_x_norm, test_x_norm) = normalization(train_x, val_x, test_x)

    # 构建数据字典
    all_data = {
        'train': {
            'x': train_x_norm,
            'target': train_target,
            'timestamp': train_timestamp,
        },
        'val': {
            'x': val_x_norm,
            'target': val_target,
            'timestamp': val_timestamp,
        },
        'test': {
            'x': test_x_norm,
            'target': test_target,
            'timestamp': test_timestamp,
        },
        'stats': {
            '_mean': stats['_mean'],
            '_std': stats['_std'],
        }
    }

    # 打印数据集信息
    print('train x:', all_data['train']['x'].shape)
    print('train target:', all_data['train']['target'].shape)
    print('train timestamp:', all_data['train']['timestamp'].shape)
    print()
    print('val x:', all_data['val']['x'].shape)
    print('val target:', all_data['val']['target'].shape)
    print('val timestamp:', all_data['val']['timestamp'].shape)
    print()
    print('test x:', all_data['test']['x'].shape)
    print('test target:', all_data['test']['target'].shape)
    print('test timestamp:', all_data['test']['timestamp'].shape)
    print()
    print('train data _mean :', stats['_mean'].shape, stats['_mean'])
    print('train data _std :', stats['_std'].shape, stats['_std'])

    # ============================================================
    import pandas as pd

    # 获取test_target
    test_target = all_data['test']['target']

    # 将test_target重新调整形状为二维（3394 * 307，12）
    reshaped_test_target = test_target.reshape(-1, test_target.shape[-1])

    # 创建列名列表
    column_names = [f'test-{i + 1}' for i in range(reshaped_test_target.shape[1])]

    # 创建DataFrame，并设置列名
    df = pd.DataFrame(reshaped_test_target, columns=column_names)

    # 写入CSV文件
    csv_filename = './test_target.csv'
    df.to_csv(csv_filename, index=False)
    # ============================================================

    # 保存数据集
    if save:
        file = os.path.basename(graph_signal_matrix_filename).split('.')[0]
        dirpath = os.path.dirname(graph_signal_matrix_filename)
        filename = os.path.join(dirpath, file + '_r' + str(num_of_hours) + '_d' + str(num_of_days) + '_w' + str(num_of_weeks)) + '_astcgn'
        print('save file:', filename)
        np.savez_compressed(filename,
                            train_x=all_data['train']['x'], train_target=all_data['train']['target'],
                            train_timestamp=all_data['train']['timestamp'],
                            val_x=all_data['val']['x'], val_target=all_data['val']['target'],
                            val_timestamp=all_data['val']['timestamp'],
                            test_x=all_data['test']['x'], test_target=all_data['test']['target'],
                            test_timestamp=all_data['test']['timestamp'],
                            mean=all_data['stats']['_mean'], std=all_data['stats']['_std']
                            )
    return all_data



def normalization(train, val, test):
    """
    对训练集、验证集和测试集进行标准化处理。

    Parameters
    ----------
    train, val, test: np.ndarray (B,N,F,T)
        待标准化的数据集，分别为训练集、验证集和测试集。

    Returns
    ----------
    stats: dict
        包含两个键值对的字典，键分别为'mean'和'std'，表示均值和标准差。
    train_norm, val_norm, test_norm: np.ndarray
        标准化后的数据集，形状与原始数据集相同。
    """

    # 确保训练集、验证集和测试集的节点数量相同
    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]
    # 计算训练集的均值，轴参数(0,1,3)表示对这些维度求均值，keepdims=True保持输出维度与输入相同
    mean = train.mean(axis=(0,1,3), keepdims=True)
    # 计算训练集的标准差，参数同上
    std = train.std(axis=(0,1,3), keepdims=True)
    # 打印均值和标准差的形状，以确保它们按预期计算
    print('mean.shape:',mean.shape)
    print('std.shape:',std.shape)

    # 定义一个内部函数normalize，用于标准化数据
    def normalize(x):
        # 将输入数据减去均值并除以标准差，实现标准化
        return (x - mean) / std

    # 使用normalize函数对训练集、验证集和测试集进行标准化
    train_norm = normalize(train)
    val_norm = normalize(val)
    test_norm = normalize(test)

    # 返回包含均值和标准差的字典，以及标准化后的数据集
    return {'_mean': mean, '_std': std}, train_norm, val_norm, test_norm



# prepare dataset
parser = argparse.ArgumentParser()
parser.add_argument("--config", default='configurations/PEMS04_astgcn.conf', type=str,
                    help="configuration file path")
args = parser.parse_args()
config = configparser.ConfigParser()
print('Read configuration file: %s' % (args.config))
config.read(args.config)
data_config = config['Data']
training_config = config['Training']

adj_filename = data_config['adj_filename']
graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
if config.has_option('Data', 'id_filename'):
    id_filename = data_config['id_filename']
else:
    id_filename = None

num_of_vertices = int(data_config['num_of_vertices'])   # 307
points_per_hour = int(data_config['points_per_hour'])   # 12
num_for_predict = int(data_config['num_for_predict'])   # 12
len_input = int(data_config['len_input'])   # 12
dataset_name = data_config['dataset_name']

num_of_weeks = int(training_config['num_of_weeks']) # 0
num_of_days = int(training_config['num_of_days'])   # 0
num_of_hours = int(training_config['num_of_hours']) # 1

graph_signal_matrix_filename = data_config['graph_signal_matrix_filename']
data = np.load(graph_signal_matrix_filename)
print(data['data'].shape)

all_data = read_and_generate_dataset(graph_signal_matrix_filename,
                                     0, 0,
                                     num_of_hours, num_for_predict,
                                     points_per_hour=points_per_hour, save=True)
