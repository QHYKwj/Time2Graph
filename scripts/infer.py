# infer_m01.py
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import torch
from config import *
from time2graph.core.model import Time2GraphWindModel
# 复用训练脚本里的气象插值函数
from scripts.run2 import load_weather_data


def load_infer_sample(base_dir,
                      weather_fname="m01_weather.csv",
                      predict_fname="m01_predict.csv",
                      true_fname="m01_true.csv",
                      step_sec=30,
                      horizon_steps=20):
    """
    使用三个文件构造推理样本：
      - predict_fname: 1 小时机舱输入 (30s 分辨率)，共 120 行
      - weather_fname: 过去12h+未来1h的逐小时气象数据（这里只用来确定时间范围）
      - true_fname:    随后10分钟(20步)的真实机舱数据，用来对比

    返回:
      time_hist:   (T_in,)  1 小时时间戳（datetime64[s]）
      turbine_1h:  (T_in, 4) = [功率, 温度, 风速, 风向]
      true_times:  (H,)     真实未来 20 步时间戳
      true_speed:  (H,)     真实未来风速
      true_dir:    (H,)     真实未来风向
    """
    # ==== 1) 读取 1 小时机舱输入 ====
    predict_path = os.path.join(base_dir, predict_fname)
    if not os.path.isfile(predict_path):
        raise FileNotFoundError(f"找不到机舱数据文件: {predict_path}")

    df_pred = pd.read_csv(predict_path)
    df_pred["time"] = pd.to_datetime(df_pred["time"])
    df_pred = df_pred.sort_values("time").reset_index(drop=True)

    input_steps = len(df_pred)
    if input_steps == 0:
        raise ValueError("m01_predict.csv 中没有任何数据")
    print(f"{predict_fname}: {input_steps} 条记录，将全部用作 1 小时输入")

    # 机舱 4 维特征
    turbine_1h = np.column_stack([
        df_pred["变频器电网侧有功功率"].values.astype(float),
        df_pred["外界温度"].values.astype(float),
        df_pred["风速"].values.astype(float),
        df_pred["风向"].values.astype(float),
    ])  # (input_steps, 4)

    # 对应时间戳 (30s)
    time_hist = df_pred["time"].astype("datetime64[s]").values  # (input_steps,)

    # ==== 2) 读取真实 10 分钟数据 ====
    true_path = os.path.join(base_dir, true_fname)
    if not os.path.isfile(true_path):
        raise FileNotFoundError(f"找不到真实数据文件: {true_path}")

    df_true = pd.read_csv(true_path)
    df_true["time"] = pd.to_datetime(df_true["time"])
    df_true = df_true.sort_values("time").reset_index(drop=True)

    total_true = len(df_true)
    if total_true < horizon_steps:
        raise ValueError(
            f"{true_fname} 里只有 {total_true} 行，不足 {horizon_steps} 个 30s 点，"
            "请确认文件内容是否为完整的 10 分钟数据。"
        )

    df_true = df_true.iloc[:horizon_steps].copy()

    # 风速 / 风向列（假设和训练集一样的中文列名）
    true_speed = df_true["风速"].values.astype(float)
    true_dir = df_true["风向"].values.astype(float)
    true_times = df_true["time"].astype("datetime64[s]").values

    return time_hist, turbine_1h, true_times, true_speed, true_dir


if __name__ == "__main__":
    base_dir = '/mnt/e/Desktop/Xplanet/AI比赛/训练集'
    farm = '风场1'
    turbine_id = 'm01'

    # 推理相关参数
    step_sec = 30          # 每一步 30s
    horizon_steps = 20     # 预测 20 步 = 10 分钟

    # 1) 读取推理样本 (1 小时输入 + m01_true 真实10分钟)
    time_hist, turbine_1h, true_times, true_speed, true_dir = load_infer_sample(
        base_dir=base_dir,
        weather_fname="m01_weather.csv",
        predict_fname="m01_predict.csv",
        true_fname="m01_true.csv",
        step_sec=step_sec,
        horizon_steps=horizon_steps
    )

    # 2) 构造 Time2GraphWindModel（一定要和训练 run2.py 的参数一致！！）
    seg_length = 24      # 保证 seg_length * num_segment == 1 小时时间点数
    num_segment = 5      # 24 * 5 = 120
    K = 50
    C = 500
    percentile = 5

    m = Time2GraphWindModel(
        K=K,
        C=C,
        seg_length=seg_length,
        num_segment=num_segment,
        init=0,
        warp=2,
        tflag=True,
        gpu_enable=True,
        percentile=percentile,
        embed_mode='concate',
        batch_size=50,
        data_size=1,
        scaled=False,
        transformer_heads=4,
        transformer_layers=2,
        transformer_ff=256,
        dropout=0.1,
        verbose=True,
        shapelets_cache='{}/scripts/cache/{}_{}_{}_{}_shapelets.cache'.format(
            module_path, 'ucr-Earthquakes', 'greedy', K, seg_length
        )
    )

    # 3) 加载已训练好的单步模型
    model_path = '{}/scripts/cache/{}/{}_embedding_t2g_model.cache'.format(
        module_path, farm, turbine_id
    )
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"找不到模型缓存文件: {model_path}")

    m.load_model(model_path)

    # 4) 自回归循环预测 20 步
    L_required = seg_length * num_segment    # 120
    T_in, _ = turbine_1h.shape
    if T_in < L_required:
        raise ValueError(
            f"输入机舱数据长度 {T_in} 小于窗口长度 {L_required}，"
            "请确认 m01_predict.csv 是否包含完整的 1 小时数据。"
        )

    # 当前“滚动窗口”的机舱数据 & 时间
    cur_turb = turbine_1h.copy()    # (>=120, 4)
    cur_times = time_hist.copy()    # (>=120,)

    # 预测结果缓存
    pred_speed_list = []
    pred_dir_list = []

    weather_path = os.path.join(base_dir, "m01_weather.csv")
    if not os.path.isfile(weather_path):
        raise FileNotFoundError(f"找不到气象数据文件: {weather_path}")

    print(f"开始自回归预测 {horizon_steps} 步...")

    for k in range(horizon_steps):
        # 取最近 L_required 个点作为输入窗口
        win_turb = cur_turb[-L_required:, :]   # (L, 4)
        win_times = cur_times[-L_required:]    # (L,)

        # 对这一段时间做气象插值 -> (L, 3)
        weather_features = load_weather_data(weather_path, win_times)

        # 拼出模型输入: (1, L, 7)
        X_win = np.concatenate([win_turb, weather_features], axis=-1)  # (L, 7)
        X_win = X_win[np.newaxis, :, :]                                # (1, L, 7)

        # 单步预测：输出 [风速, 风向]，形状 (1, 2)
        y_step = m.predict(X_win)[0]   # (2,)
        v_pred = float(y_step[0])
        dir_pred = float(y_step[1])

        pred_speed_list.append(v_pred)
        pred_dir_list.append(dir_pred)

        # 生成下一个时刻的时间戳（30s 后）
        new_time = cur_times[-1] + np.timedelta64(step_sec, 's')

        # 新的机舱点：功率 / 温度沿用上一时刻，风速 / 风向用预测
        last_power = cur_turb[-1, 0]
        last_temp  = cur_turb[-1, 1]
        new_point = np.array([[last_power, last_temp, v_pred, dir_pred]])  # (1, 4)

        # 追加到序列末尾，为下一步预测准备
        cur_turb = np.vstack([cur_turb, new_point])
        cur_times = np.append(cur_times, new_time)

    pred_speed = np.array(pred_speed_list)   # (20,)
    pred_dir   = np.array(pred_dir_list)     # (20,)

    # 5) 组织输出表格
    # time 列：优先用 true_times（真实数据时间），长度不对再用预测生成的时间
    if len(true_times) == horizon_steps:
        time_col = true_times
    else:
        # 如果 true_times 长度不对，就用 cur_times 中最后 horizon_steps 个
        time_col = cur_times[-horizon_steps:]

    speed_err = pred_speed - true_speed
    dir_err   = pred_dir - true_dir

    result_df = pd.DataFrame({
        "time": pd.to_datetime(time_col),
        "真实风速": true_speed,
        "预测风速": pred_speed,
        "风速误差": speed_err,
        "真实风向": true_dir,
        "预测风向": pred_dir,
        "风向误差": dir_err,
    })

    out_path = os.path.join(base_dir, "m01_predict_result.csv")
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("预测结果已保存到:", out_path)
    print(result_df.head())
