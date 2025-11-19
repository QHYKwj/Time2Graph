# -*- coding: utf-8 -*-
import sys
import time
import itertools
import psutil
from subprocess import *
import numpy as np  # 添加 numpy 导入

class ModelUtils(object):
    """
        model utils for basic classifiers.
        kwargs list:
            lr paras
                penalty: list of str, candidate: l1, l2;
                c: list of float
                inter_scale: list of float
            rf and dts paras:
                criteria: list of str, candidate: gini, entropy
                max_features: list of str(including None), candidate: auto, log2 or None
                max_depth: list of int
                max_split: list of int
                min_leaf: list of int
            xgb paras:
                max_depth: list of int
                learning_rate: list of float
                n_jobs: int
                class_weight: list of int
                booster: list of str, candidate: gblinear, gbtree, dart
            svm paras:
                c: list of float
                svm_kernel: list of str, candidate: rbf, poly, sigmoid
            deepwalk paras:
                num_walks: list of int
                representation_size: list of int
                window_size: list of int
                workers: int
                undirected: bool
    """
    def __init__(self, kernel, **kwargs):
        self.kernel = kernel
        self.kwargs = kwargs

    @property
    def clf__(self):
        if self.kernel == 'lr':
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression
        elif self.kernel == 'svm':
            # from sklearn.svm import SVC
            # return SVC
            from sklearn.svm import SVR  # 使用支持回归的支持向量机
            return SVR
        elif self.kernel == 'dts':
            # from sklearn.tree import DecisionTreeClassifier
            from sklearn.tree import DecisionTreeRegressor  # 使用回归树
            return DecisionTreeRegressor
        elif self.kernel == 'rf':
            from sklearn.ensemble import RandomForestRegressor  # 使用回归森林
            # from sklearn.ensemble import RandomForestClassifier
            return RandomForestRegressor
        elif self.kernel == 'xgb':
            # from xgboost import XGBClassifier
            # return XGBClassifier
            from xgboost import XGBRegressor # 使用XGB回归
            return XGBRegressor
        else:
            raise NotImplementedError('unsupported kernel {}'.format(self.kernel))

    def para_len(self, balanced):
        cnt = 0
        for _ in self.clf_paras(balanced=balanced):
            cnt += 1
        return cnt

    def clf_paras(self, balanced):
        class_weight = 'balanced' if balanced else None
        if self.kernel == 'lr':
            penalty = self.kwargs.get('penalty', ['l1', 'l2'])
            c = self.kwargs.get('c', [pow(5, i) for i in range(-3, 3)])
            intercept_scaling = self.kwargs.get('inter_scale', [pow(5, i) for i in range(-3, 3)])
            for (p1, p2, p3) in itertools.product(penalty, c, intercept_scaling):
                yield {
                    'penalty': p1,
                    'C': p2,
                    'intercept_scaling': p3,
                    # 'class_weight': class_weight
                }
        elif self.kernel == 'rf' or self.kernel == 'dts':
            criteria = self.kwargs.get('criteria', ['gini', 'entropy'])
            max_features = self.kwargs.get('max_feature', ['auto', 'log2',  None])
            max_depth = self.kwargs.get('max_depth', [10, 25, 50])
            min_samples_split = self.kwargs.get('max_split', [2, 4, 8])
            min_samples_leaf = self.kwargs.get('min_leaf', [1, 3, 5])
            for (p1, p2, p3, p4, p5) in itertools.product(
                    criteria, max_features, max_depth, min_samples_split, min_samples_leaf
            ):
                yield {
                    'criterion': p1,
                    'max_features': p2,
                    'max_depth': p3,
                    'min_samples_split': p4,
                    'min_samples_leaf': p5,
                    # 'class_weight': class_weight
                }
        # elif self.kernel == 'xgb':
        #     max_depth = self.kwargs.get('max_depth', [1, 2, 4, 8, 12, 16])
        #     learning_rate = self.kwargs.get('learning_rate', [0.1, 0.2, 0.3])
        #     n_jobs = [self.kwargs.get('n_jobs', psutil.cpu_count())]
        #     class_weight = self.kwargs.get('class_weight', [1, 10, 50, 100])
        #     booster = self.kwargs.get('booster', ['gblinear', 'gbtree', 'dart'])
        #     for (p1, p2, p3, p4, p5) in itertools.product(
        #             max_depth, learning_rate, booster, n_jobs, class_weight
        #     ):
        #         yield {
        #             'max_depth': p1,
        #             'learning_rate': p2,
        #             'booster': p3,
        #             'n_jobs': p4,
        #             'scale_pos_weight': p5
        #         }
        elif self.kernel == 'xgb':
            max_depth = self.kwargs.get('max_depth', [4])          # 原来 6 个 → 1 个
            learning_rate = self.kwargs.get('learning_rate', [0.1])# 原来 3 个 → 1 个
            n_jobs = [self.kwargs.get('n_jobs', psutil.cpu_count())]
            class_weight = self.kwargs.get('class_weight', [1])    # 原来 4 个 → 1 个
            booster = self.kwargs.get('booster', ['gbtree'])       # 原来 3 个 → 1 个

            for (p1, p2, p3, p4, p5) in itertools.product(
                    max_depth, learning_rate, booster, n_jobs, class_weight
            ):
                yield {
                    'max_depth': p1,
                    'learning_rate': p2,
                    'booster': p3,
                    'n_jobs': p4,
                    # 'scale_pos_weight': p5
                }
        elif self.kernel == 'svm':
            c = self.kwargs.get('c', [pow(2, i) for i in range(-2, 2)])
            svm_kernel = self.kwargs.get('svm_kernel', ['rbf', 'poly', 'sigmoid'])
            for (p1, p2) in itertools.product(c, svm_kernel):
                yield {
                    'C': p1,
                    'kernel': p2,
                    # 'class_weight': class_weight
                    }
        else:
            raise NotImplementedError()

    @staticmethod
    def partition_data__(data, ratio, shuffle=True, multi=True):
        import random
        if not multi:
            size = len(data)
            if shuffle:
                idx = random.sample(range(size), int(size * ratio))
            else:
                idx, step, cnt, init = [], 1.0 / ratio, 0, 0
                while cnt < int(size * ratio):
                    idx.append(int(init))
                    init += step
            return data[idx]
        else:
            num, size = len(data), len(data[0])
            if shuffle:
                idx = random.sample(range(size), int(size * ratio))
            else:
                idx, step, cnt, init = [], 1.0 / ratio, 0, 0
                while cnt < int(size * ratio):
                    idx.append(int(init))
                    init += step
            return [data[k][idx] for k in range(num)]

    def deepwalk_paras(self):
        num_walks = self.kwargs.get('num_walks', [10, 20])
        representation_size = self.kwargs.get('representation_size', [32, 64, 128, 256])
        walk_length = self.kwargs.get('walk_length', [32, 64, 128])
        window_size = self.kwargs.get('window_size', [5, 10])
        workers = self.kwargs.get('workers', psutil.cpu_count())
        undirected = self.kwargs.get('undirected', False)
        for (p1, p2, p3, p4) in itertools.product(
                num_walks, representation_size, walk_length, window_size
        ):
            yield {
                'number-walks': p1,
                'representation-size': p2,
                'walk-length': p3,
                'window-size': p4,
                'workers': workers,
                'undirected': undirected
            }

    def return_metric_method(self, opt_metric):
        # from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        # if opt_metric == 'accuracy':
        #     return accuracy_score
        # elif opt_metric == 'precision':
        #     return precision_score
        # elif opt_metric == 'recall':
        #     return recall_score
        # elif opt_metric == 'f1':
        #     return f1_score
        # else:
        #     raise NotImplementedError('unsupported metric {}'.format(opt_metric))
        from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
        if opt_metric == 'mse':  # 均方误差
            return mean_squared_error
        elif opt_metric == 'rmse':
            # RMSE = sqrt(MSE)
            return lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
        elif opt_metric == 'r2':  # 决定系数
            return r2_score
        elif opt_metric == 'mae':
            return mean_absolute_error
        else:
            raise NotImplementedError('unsupported metric {}'.format(opt_metric))

    def transformer_regressor_paras(self):
        """
        为 Time2GraphWindModel 的 Transformer 部分生成参数
        """
        num_heads = self.kwargs.get('num_heads', [4, 8])
        num_layers = self.kwargs.get('num_layers', [2, 4])
        ff_hidden_dim = self.kwargs.get('ff_hidden_dim', [256, 512])
        dropout = self.kwargs.get('dropout', [0.1, 0.2, 0.3])
        lr = self.kwargs.get('lr', [1e-3, 1e-4])
        batch_size = self.kwargs.get('batch_size', [32, 64])
        
        for (p1, p2, p3, p4, p5, p6) in itertools.product(
                num_heads, num_layers, ff_hidden_dim, dropout, lr, batch_size
        ):
            yield {
                'num_heads': p1,
                'num_layers': p2,
                'ff_hidden_dim': p3,
                'dropout': p4,
                'lr': p5,
                'batch_size': p6
            }

    def load_model(self, fpath, **kwargs):
        pass

    def save_model(self, fpath, **kwargs):
        pass

    def fit(self, X, Y, **kwargs):
        pass

    def predict(self, X, **kwargs):
        pass


class Debugger(object):
    """
        Class for debugger print
    """
    def __init__(self):
        pass

    @staticmethod
    def error_print(msg, debug=True):
        if debug:
            print('[error]' + msg)

    @staticmethod
    def warn_print(msg, debug=True):
        if debug:
            print('[warning]' + msg)

    @staticmethod
    def debug_print(msg, debug=True):
        if debug:
            print('[debug]' + msg + '\r', end='')
            sys.stdout.flush()

    @staticmethod
    def info_print(msg):
        print('[info]' + msg)

    @staticmethod
    def time_print(msg, begin, profiling=False):
        if profiling:
            assert isinstance(begin, type(time.time())), 'invalid begin time {}'.format(begin)
            print('[info]{}, elapsed for {:.2f}s'.format(msg, time.time() - begin))


class Queue:
    def __init__(self, max_size):
        self.queue = []
        self.max_size = max_size

    def enqueue(self, val):
        if self.size() == self.max_size:
            self.dequeue()
        self.queue.insert(0, val)

    def dequeue(self):
        if self.is_empty():
            return None
        else:
            return self.queue.pop()

    def size(self):
        return len(self.queue)

    def is_empty(self):
        return self.size() == 0


def convert_string(string, val, cvt_type='float'):
    """
        Convert a string as given type.
    :param string:  input string
    :param val: default return value if conversion fails
    :param cvt_type: conversion type
    :return: value with given type
    """
    try:
        return eval(cvt_type)(string)
    except NameError as _:
        Debugger.warn_print('invalid convert type {}; use float() by default'.format(cvt_type))
        return float(string)
    except ValueError as _:
        Debugger.warn_print('invalid convert value {}; return {} by default'.format(string, val))
        return val


def syscmd(cmd, encoding=''):
    """
        Runs a command on the system, waits for the command to finish, and then
    returns the text output of the command. If the command produces no text
    output, the command's return code will be returned instead.

    :param cmd: command, str
    :param encoding: encoding method, str(utf8, unicode, etc)
    :return: return code or text output
    """
    p = Popen(cmd, shell=True, stdin=PIPE, stdout=PIPE,
              stderr=STDOUT, close_fds=True)
    p.wait()
    output = p.stdout.read()
    if len(output) > 1:
        if encoding:
            return output.decode(encoding)
        else:
            return output
    return p.returncode
