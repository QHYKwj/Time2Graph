# -*- coding: utf-8 -*-
import argparse
import warnings
import os
from config import *
from time2graph.utils.base_utils import Debugger
from time2graph.core.model import Time2GraphWindModel,ShapeletSeqRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from datetime import datetime

# 1) 加载一个风机的所有 csv：返回 (data, timestamps)
# data: (T, 4) -> [power, temperature, wind_speed, wind_direction]
# timestamps: (T,) -> numpy datetime64[s]
def load_turbine_data(turbine_dir):
    turbine_data_list = []
    ts_list = []

    files = sorted([f for f in os.listdir(turbine_dir) if f.endswith('.csv')])
    if not files:
        raise ValueError(f"风机目录下没有 csv 文件: {turbine_dir}")

    for file in files:
        file_path = os.path.join(turbine_dir, file)
        df = pd.read_csv(file_path)

        # 必须字段
        required_cols = ["time", "变频器电网侧有功功率", "外界温度", "风向", "风速"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"⚠ 跳过文件 {file_path}，缺少列: {missing}")
            continue

        # 只保留需要的列，避免其他乱七八糟的 NaN 干扰
        df = df[required_cols].copy()

        # 把 inf/-inf 先变成 NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # 用前向填充 + 后向填充补上缺失（避免直接 drop 掉时间点）
        df["变频器电网侧有功功率"].interpolate(inplace=True)
        df["外界温度"].interpolate(inplace=True)
        df["风速"].interpolate(inplace=True)
        df["风向"].interpolate(inplace=True)

        df.fillna(method="ffill", inplace=True)
        df.fillna(method="bfill", inplace=True)

        # 清洗之后如果还是全 NaN，就跳过这个文件
        if df[["变频器电网侧有功功率", "外界温度", "风速", "风向"]].isna().all().any():
            print(f"⚠ 清洗后仍有关键列全为 NaN，跳过文件 {file_path}")
            continue

        # 时间
        time_stamps = pd.to_datetime(df["time"]).astype("datetime64[s]").values

        power = df["变频器电网侧有功功率"].values
        temperature = df["外界温度"].values
        wind_speed = df["风速"].values
        wind_dir = df["风向"].values

        data = np.column_stack([power, temperature, wind_speed, wind_dir])

        turbine_data_list.append(data)
        ts_list.append(time_stamps)

    if not turbine_data_list:
        raise ValueError(f"❌ {turbine_dir} 下所有 csv 都被跳过，检查列名/数据。")

    turbine_data = np.concatenate(turbine_data_list, axis=0)   # (T, 4)
    timestamps = np.concatenate(ts_list, axis=0)               # (T,)

    print(f"✔ {turbine_dir} 加载完成: 数据点 {turbine_data.shape[0]}")
    return turbine_data, timestamps


# 2) 加载天气数据并按机舱时间插值
# 输入: weather_file, target_time_stamps (datetime64)
# 输出: (T, 3) -> [wind_spd, cosθ, sinθ]
def load_weather_data(weather_file, target_time_stamps):
    weather_df = pd.read_csv(weather_file)
    # 清洗 NaN / inf
    weather_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # 时间 → 秒
    weather_time_dt = pd.to_datetime(weather_df["time"])
    weather_time = weather_time_dt.astype("int64") // 10**9  # int64 秒

    # 风速列
    if "wind_spd" in weather_df.columns:
        ws = weather_df["wind_spd"]
    elif "风速" in weather_df.columns:
        ws = weather_df["风速"]
    else:
        raise KeyError(f"{weather_file} 中找不到 wind_spd / 风速 列")

    # 风向列
    if "wind_dir" in weather_df.columns:
        wd = weather_df["wind_dir"]
    elif "风向" in weather_df.columns:
        wd = weather_df["风向"]
    else:
        raise KeyError(f"{weather_file} 中找不到 wind_dir / 风向 列")
    
    # 插值 + 前后填充，补齐 NaN
    ws = ws.astype(float)
    wd = wd.astype(float)

    ws.interpolate(inplace=True)
    wd.interpolate(inplace=True)
    ws.fillna(method="ffill", inplace=True)
    ws.fillna(method="bfill", inplace=True)
    wd.fillna(method="ffill", inplace=True)
    wd.fillna(method="bfill", inplace=True)

    weather_wind_speed = ws.values
    weather_wind_dir = wd.values

    # 目标时间 → 秒
    target_ts = target_time_stamps.astype("datetime64[s]").astype("int64")

    # 一维线性插值（自变量是秒数）
    speed_interp = interp1d(
        weather_time,
        weather_wind_speed,
        kind="linear",
        fill_value="extrapolate",
        assume_sorted=True,
    )
    dir_interp = interp1d(
        weather_time,
        weather_wind_dir,
        kind="linear",
        fill_value="extrapolate",
        assume_sorted=True,
    )

    interpolated_speed = speed_interp(target_ts)      # (T,)
    interpolated_dir = dir_interp(target_ts)          # (T,)

    # 角度 → cos, sin
    rad = np.deg2rad(interpolated_dir)
    cos_dir = np.cos(rad)
    sin_dir = np.sin(rad)

    weather_features = np.column_stack([interpolated_speed, cos_dir, sin_dir])  # (T, 3)
    return weather_features


# 3) 按时间切 train / test，并拼接机舱 + 气象，生成 X, y
# X_train: (T_train, 7)  = [power,temp,wind_spd,wind_dir, met_spd, met_cos, met_sin]
# y_train: (T_train, 4)  = x_train 向后 roll 20 步（未来10分钟的机舱 4 维）
def load_full_dataset(base_dir, farm, turbine_id, train_end_date):
    """
    base_dir: /mnt/e/Desktop/Xplanet/AI比赛/训练集
    farm: '风场1' or '风场2'
    turbine_id: 'm01' ... 'm50'
    train_end_date: '2019-06-30'
    """
    turbine_dir = os.path.join(base_dir, farm, turbine_id)
    weather_file = os.path.join(base_dir, farm, "weather.csv")

    # 1) 加载整个风机两年数据 + 时间戳
    turbine_data, time_stamps = load_turbine_data(turbine_dir)   # (T,4), (T,)

    # 2) 加载同一时间轴上的气象特征
    weather_data = load_weather_data(weather_file, time_stamps)  # (T,3)

    # 3) 拼接机舱 + 气象 → 特征序列
    full_data = np.concatenate([turbine_data, weather_data], axis=-1)  # (T,7)

    # 4) 时间切 train/test
    train_end_ts = pd.to_datetime(train_end_date)   # pandas Timestamp

    train_mask = time_stamps <= train_end_ts
    test_mask = time_stamps > train_end_ts

    x_train = full_data[train_mask]   # (T_train, 7)
    x_test  = full_data[test_mask]    # (T_test, 7)

    # 对应的机舱 4 维，用于生成回归标签 y（未来10分钟）
    turbine_train = turbine_data[train_mask]  # (T_train, 4)
    turbine_test  = turbine_data[test_mask]   # (T_test, 4)

    # 5) 生成回归目标：用 np.roll 取未来 20 步（10 分钟）的机舱 4 维
    # 注意：最后 20 个样本的标签会“循环”到前面，如果要严格避免泄露，可以后续再裁剪。
    horizon = 1
    y_train = np.roll(turbine_train, -horizon, axis=0)  # (T_train, 4)
    y_test  = np.roll(turbine_test, -horizon, axis=0)   # (T_test, 4)

    print(f"{farm}/{turbine_id} -> x_train: {x_train.shape}, y_train: {y_train.shape}, "
          f"x_test: {x_test.shape}, y_test: {y_test.shape}")
    
    # === debug 1: 检查 np.roll 的效果是不是 +1 步 ===
    print("=== debug roll in load_full_dataset ===")
    T_train = turbine_train.shape[0]
    check_idx = [100, 500, 1000]  # 可以随便挑几个，不越界就行
    for t in check_idx:
        if t + 1 >= T_train:
            continue
        cur_speed = turbine_train[t, 2]
        next_speed = turbine_train[t + 1, 2]
        rolled_speed = y_train[t, 2]
        print(f"t={t}: cur_speed={cur_speed:.4f}, "
              f"next_speed={next_speed:.4f}, "
              f"rolled_label_speed={rolled_speed:.4f}")
    
    return x_train, y_train, x_test, y_test

def build_t2g_dataset(x_seq, y_seq, seg_length, num_segment):
    """
    x_seq: (T, D)  长时间序列特征
    y_seq: (T, Fy) 长时间序列标签（这里 Fy=4，对应机舱4个变量）
    seg_length, num_segment: 来自命令行，比如 24 * 21 = 504
    target_col: y_seq 里要预测的那一列（默认 2 = 风速）
    
    返回:
        X: (N, L, D)
        y: (N,)
    """
    L = seg_length * num_segment          # 每个样本的时间长度
    T = x_seq.shape[0]
    D = x_seq.shape[1]
    Fy = y_seq.shape[1]

    if T < L:
        raise ValueError(f"序列长度 {T} 小于一个样本长度 {L}，无法切片。")

    # 为了简单，先用“非重叠窗口”：从头开始每 L 个点切一个样本
    N = T // L
    T_trim = N * L    # 截断到可以整除的长度

    x_seq = x_seq[:T_trim]         # (T_trim, D)
    y_seq = y_seq[:T_trim]         # (T_trim, Fy)

    # 切成 (N, L, D)
    X = x_seq.reshape(N, L, D)     # (N, L, D)
    y_block = y_seq.reshape(N, L, Fy)  # (N, L, Fy)

    # 取每段最后一个时间点的风速和风向作为回归标签
    # 机舱数据中：第2列是风速，第3列是风向
    wind_speed = y_block[:, -1, 2]     # (N,)
    wind_direction = y_block[:, -1, 3] # (N,)
    
    Y = np.column_stack([wind_speed, wind_direction])  # (N, 2)

    # === debug 2: 检查窗口最后一个输入点 & 标签的关系 ===
    print("=== debug window-level label in build_t2g_dataset ===")
    print("X shape:", X.shape, "Y shape:", Y.shape)  # 期望: (N, L, 7), (N, 2)
    for i in range(min(5, N)):
        last_speed_in_X = X[i, -1, 2]
        last_dir_in_X   = X[i, -1, 3]
        label_speed, label_dir = Y[i]
        print(
            f"sample {i}: last_speed_in_X={last_speed_in_X:.4f}, "
            f"label_speed(+1 step)={label_speed:.4f} | "
            f"last_dir_in_X={last_dir_in_X:.2f}, "
            f"label_dir(+1 step)={label_dir:.2f}"
        )

    return X, Y

def filter_invalid_samples(X, Y):
    """
    X: (N, L, D)
    Y: (N, 2)  # 两个目标：风速和风向
    过滤掉任何在 X 或 Y 中包含 NaN/inf 的样本
    """
    # X 中每个样本是否全部是有限值
    mask_X = np.isfinite(X).all(axis=(1, 2))  # (N, )
    
    # Y 中每个样本是否全部是有限值（注意：这里只对 Y 中的每个样本进行检查）
    mask_Y = np.isfinite(Y).all(axis=1)  # (N, )，检查每个样本的所有目标值（风速和风向）

    mask = mask_X & mask_Y  # 需要保证 X 和 Y 都有效

    X_clean = X[mask]
    Y_clean = Y[mask]

    print(f"过滤无效样本: 原 {len(Y)} -> 有效 {len(Y_clean)}")
    return X_clean, Y_clean

if __name__ == '__main__':
    warnings.filterwarnings(module='sklearn*', action='ignore', category=DeprecationWarning)
    
    # 命令行参数解析部分
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='ucr-Earthquakes',
                        help='ucr-Earthquakes/WormsTwoClass/Strawberry')
    parser.add_argument('--K', type=int, default=100, help='number of shapelets extracted')
    parser.add_argument('--C', type=int, default=800, help='number of shapelet candidates')
    parser.add_argument('--n_splits', type=int, default=2, help='number of splits in cross-validation')
    parser.add_argument('--num_segment', type=int, default=12, help='number of segment a time series is divided into')
    parser.add_argument('--seg_length', type=int, default=30, help='segment length')
    parser.add_argument('--njobs', type=int, default=8, help='number of threads in parallel')
    parser.add_argument('--data_size', type=int, default=1, help='data dimension of time series')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer used in time-aware shapelets learning')
    parser.add_argument('--alpha', type=float, default=0.1, help='penalty parameter of local timing factor')
    parser.add_argument('--beta', type=float, default=0.05, help='penalty parameter of global timing factor')
    parser.add_argument('--init', type=int, default=0, help='init index of time series data')
    parser.add_argument('--gpu_enable', action='store_true', default=False, help='bool, whether to use GPU')
    parser.add_argument('--opt_metric', type=str, default='mae',
                        help='which metric to optimize in prediction (rmse/mse/r2)')

    parser.add_argument('--cache', action='store_true', default=False,
                        help='whether to dump model to local file')
    parser.add_argument('--embed', type=str, default='concate',
                        help='which embed strategy to use (aggregate/concate)')
    parser.add_argument('--embed_size', type=int, default=256, help='embedding size of shapelets')
    parser.add_argument('--warp', type=int, default=2, help='warping size in greedy-dtw')
    parser.add_argument('--cmethod', type=str, default='greedy',
                        help='which algorithm to use in candidate generation (cluster/greedy)')
    parser.add_argument('--kernel', type=str, default='xgb', help='specify outer-classifier (default xgboost)')
    parser.add_argument('--percentile', type=int, default=10,
                        help='percentile for distance threshold in constructing graph')
    parser.add_argument('--measurement', type=str, default='gdtw',
                        help='which distance metric to use (default greedy-dtw)')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='batch size in each training step')
    parser.add_argument('--tflag', action='store_false', default=True,
                        help='flag that whether to use timing factors')
    parser.add_argument('--scaled', action='store_true', default=False,
                        help='flag that whether to rescale time series data')
    parser.add_argument('--norm', action='store_true', default=False,
                        help='flag that whether to normalize extracted representations')
    parser.add_argument('--no_global', action='store_false', default=True,
                        help='whether to use global timing factors')

    # 新增：指定要跑的风场和风机
    parser.add_argument('--farm', type=str, default='风场1',
                        help='wind farm name, e.g. 风场1 or 风场2')
    parser.add_argument('--turbine_id', type=str, default='m01',
                        help='turbine id, e.g. m01, m02, ...')

    # 新增 Transformer 相关参数
    parser.add_argument('--transformer_heads', type=int, default=4,
                        help='number of attention heads in Transformer')
    parser.add_argument('--transformer_layers', type=int, default=2,
                        help='number of Transformer layers')
    parser.add_argument('--transformer_ff', type=int, default=256,
                        help='hidden dimension in Transformer FFN')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout rate')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='number of training epochs')
    parser.add_argument('--regressor_batch_size', type=int, default=64,
                        help='batch size for regressor training')

    args = parser.parse_args()
    Debugger.info_print('running with {}'.format(args.__dict__))
    
    # 数据目录路径
    base_dir = '/mnt/e/Desktop/Xplanet/AI比赛/训练集'  # 根据您的数据目录调整
    
    # 设定每个风机的训练集结束日期（2019年6月30日）


    # -------- 只跑一个风机：由参数控制，默认 风场1 / m01 --------
    farm = args.farm
    turbine_id = args.turbine_id
    if farm=='风场1':
        train_end_date = "2019-06-30"
    else:
        train_end_date = "2018-06-30"
    print(f"正在训练 {farm}/{turbine_id} 的模型...")

    # 1) 加载该风机的训练数据和测试数据
    x_train_seq, y_train_seq, x_test_seq, y_test_seq = load_full_dataset(
        base_dir, farm, turbine_id, train_end_date
    )

    # 2) 切成 Time2Graph 需要的 (N, L, D)
    X_train, Y_train = build_t2g_dataset(
        x_train_seq, y_train_seq,
        seg_length=args.seg_length,
        num_segment=args.num_segment
    )
    X_test, Y_test = build_t2g_dataset(
        x_test_seq, y_test_seq,
        seg_length=args.seg_length,
        num_segment=args.num_segment
    )

    # 3) 过滤 NaN / inf
    X_train, Y_train = filter_invalid_samples(X_train, Y_train)
    X_test, Y_test   = filter_invalid_samples(X_test, Y_test)
    print(f"{farm}/{turbine_id} -> X_train: {X_train.shape}, Y_train: {Y_train.shape}, "
          f"X_test: {X_test.shape}, Y_test: {Y_test.shape}")
    # === debug 3: 再次确认窗口级输入 / 标签 ===
    print("=== debug 3: final train set check ===")
    for i in range(min(5, X_train.shape[0])):
        last_speed = X_train[i, -1, 2]
        last_dir   = X_train[i, -1, 3]
        label_speed, label_dir = Y_train[i]
        print(
            f"[final] sample {i}: last_speed_in_X={last_speed:.4f}, "
            f"label_speed(+1)={label_speed:.4f} | "
            f"last_dir_in_X={last_dir:.2f}, "
            f"label_dir(+1)={label_dir:.2f}"
        )
    # 4) 创建模型
    m = Time2GraphWindModel(
        K=args.K,
        C=args.C,
        seg_length=args.seg_length,
        num_segment=args.num_segment,
        init=args.init,
        warp=args.warp,
        tflag=args.tflag,
        gpu_enable=args.gpu_enable,
        percentile=args.percentile,
        embed_mode=args.embed,  # 必须是 'concate'
        batch_size=args.batch_size,
        data_size=args.data_size,
        scaled=args.scaled,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        transformer_ff=args.transformer_ff,
        dropout=args.dropout,
        verbose=True,
        shapelets_cache='{}/scripts/cache/{}/{}_{}_{}_{}_shapelets.cache'.format(
            module_path, farm ,turbine_id ,args.cmethod, args.K, args.seg_length)
    )

    # 5) 训练并保存模型（缓存路径带上风场和风机）
    cache_dir = '{}/scripts/cache/{}/{}'.format(module_path, farm, turbine_id)
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)

    print("NaN in X_train:", np.isnan(X_train).sum(),
          "NaN in Y_train:", np.isnan(Y_train).sum())
    print("Inf in X_train:", np.isinf(X_train).sum(),
          "Inf in Y_train:", np.isinf(Y_train).sum())

    print("开始训练模型...")
    m.fit(X=X_train,
          Y=Y_train,
          lr=args.lr,
          num_epochs=args.num_epochs,
          batch_size=args.regressor_batch_size,
          cache_dir=cache_dir)

    # 无论 args.cache 与否，都缓存这个风机的模型
    model_path = '{}/scripts/cache/{}/{}_embedding_t2g_model.cache'.format(
        module_path, farm, turbine_id
    )
    m.save_model(fpath=model_path)
    Debugger.info_print(f"模型已保存到: {model_path}")

    # 6) 预测结果
    # print("开始预测...")
    # y_pred = m.predict(X_test)  # (N_test, 2)
    print("开始预测（分批）...")
    all_preds = []
    batch = 256
    N = X_test.shape[0]
    for i in range(0, N, batch):
        x_batch = X_test[i:i+batch]
        preds = m.predict(x_batch)
        all_preds.append(preds)
    y_pred = np.vstack(all_preds)
    # 计算风速的 RMSE 和 MAE
    wind_speed_pred = y_pred[:, 0]  # 假设风速是预测的第一个列
    wind_speed_true = Y_test[:, 0]
    wind_speed_mse = mean_squared_error(wind_speed_true, wind_speed_pred)
    wind_speed_mae = mean_absolute_error(wind_speed_true, wind_speed_pred)
    wind_speed_rmse = np.sqrt(wind_speed_mse)
    
    # 计算风向的 MAE
    wind_direction_pred = y_pred[:, 1]  # 假设风向是预测的第二个列
    wind_direction_true = Y_test[:, 1]
    wind_direction_mae = mean_absolute_error(wind_direction_true, wind_direction_pred)
    
    # 打印结果
    Debugger.info_print(
        f"{farm}/{turbine_id} 的结果: "
        f"风速 RMSE: {wind_speed_rmse:.4f}, 风速 MSE: {wind_speed_mse:.4f}, "
        f"风速 MAE: {wind_speed_mae:.4f}, 风向 MAE: {wind_direction_mae:.4f}"
    )


