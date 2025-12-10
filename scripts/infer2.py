# infer_m01.py
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import torch
from config import *
from time2graph.core.model import Time2GraphWindModel
import math

# 复用训练脚本里的气象插值函数（需要更新以返回Cos/Sin）
def load_weather_data(weather_file, target_time_stamps):
    """
    从weather_file读取气象数据，并对target_time_stamps进行线性插值
    返回: (T, 3) -> [wind_spd, cosθ, sinθ]
    """
    df_weather = pd.read_csv(weather_file)
    df_weather["time"] = pd.to_datetime(df_weather["time"])
    
    # 提取气象场的风速风向
    weather_spd = df_weather["wind_spd"].values.astype(float)
    weather_dir_deg = df_weather["wind_dir"].values.astype(float)
    
    # 将风向转为 Cos/Sin 表示
    weather_dir_rad = np.deg2rad(weather_dir_deg)
    weather_cos = np.cos(weather_dir_rad)
    weather_sin = np.sin(weather_dir_rad)
    
    weather_times = df_weather["time"].astype("datetime64[s]").values
    
    # 对每个目标时间戳进行线性插值
    result = []
    for t in target_time_stamps:
        # 找到最近的两个气象时间点
        idx = np.searchsorted(weather_times, t)
        
        if idx == 0:
            # 目标时间在第一个气象点之前，使用第一个点
            spd = weather_spd[0]
            cos_val = weather_cos[0]
            sin_val = weather_sin[0]
        elif idx == len(weather_times):
            # 目标时间在最后一个气象点之后，使用最后一个点
            spd = weather_spd[-1]
            cos_val = weather_cos[-1]
            sin_val = weather_sin[-1]
        else:
            # 在两个气象点之间进行线性插值
            t_prev = weather_times[idx-1]
            t_next = weather_times[idx]
            
            # 计算时间权重
            if t_next == t_prev:
                weight = 0.5
            else:
                weight = float((t - t_prev) / (t_next - t_prev))
            
            # 风速线性插值
            spd_prev = weather_spd[idx-1]
            spd_next = weather_spd[idx]
            spd = spd_prev + weight * (spd_next - spd_prev)
            
            # Cos/Sin 角度插值需要特殊处理
            # 先对角度进行插值，再转回Cos/Sin
            dir_prev = np.arctan2(weather_sin[idx-1], weather_cos[idx-1])
            dir_next = np.arctan2(weather_sin[idx], weather_cos[idx])
            
            # 处理角度跨越0/2π边界的情况
            if abs(dir_next - dir_prev) > np.pi:
                if dir_next > dir_prev:
                    dir_prev += 2 * np.pi
                else:
                    dir_next += 2 * np.pi
            
            dir_interp = dir_prev + weight * (dir_next - dir_prev)
            dir_interp = dir_interp % (2 * np.pi)
            
            cos_val = np.cos(dir_interp)
            sin_val = np.sin(dir_interp)
        
        result.append([spd, cos_val, sin_val])
    
    return np.array(result)


def load_infer_sample(base_dir,
                      weather_fname="weather.csv",
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
      turbine_1h:  (T_in, 5) = [功率, 温度, 风速, Cosθ, Sinθ]
      true_times:  (H,)     真实未来 20 步时间戳
      true_speed:  (H,)     真实未来风速
      true_cos:    (H,)     真实未来风向Cos值
      true_sin:    (H,)     真实未来风向Sin值
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

    # 将风向转为 Cos/Sin 表示
    wind_dir_deg = df_pred["风向"].values.astype(float)
    wind_dir_rad = np.deg2rad(wind_dir_deg)
    wd_cos = np.cos(wind_dir_rad)
    wd_sin = np.sin(wind_dir_rad)

    # 机舱 5 维特征：功率, 温度, 风速, Cosθ, Sinθ
    turbine_1h = np.column_stack([
        df_pred["变频器电网侧有功功率"].values.astype(float),
        df_pred["外界温度"].values.astype(float),
        df_pred["风速"].values.astype(float),
        wd_cos,
        wd_sin,
    ])  # (input_steps, 5)

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

    # 提取真实风速和风向（转为Cos/Sin）
    true_speed = df_true["风速"].values.astype(float)
    true_dir_deg = df_true["风向"].values.astype(float)
    true_dir_rad = np.deg2rad(true_dir_deg)
    true_cos = np.cos(true_dir_rad)
    true_sin = np.sin(true_dir_rad)
    true_times = df_true["time"].astype("datetime64[s]").values

    return time_hist, turbine_1h, true_times, true_speed, true_cos, true_sin


def convert_cos_sin_to_degrees(cos_vals, sin_vals):
    """
    将Cos/Sin值转换回角度（0-360度）
    """
    angles_rad = np.arctan2(sin_vals, cos_vals)
    angles_deg = np.rad2deg(angles_rad)
    angles_deg = angles_deg % 360
    return angles_deg


if __name__ == "__main__":
    # base_dir = '/mnt/e/Desktop/Xplanet/AI比赛/训练集'
    base_dir = '/mnt/e/Desktop/Xplanet/AI比赛/预测数据/m01'
    farm = '风场1'
    turbine_id = 'm01'

    # 推理相关参数
    step_sec = 30          # 每一步 30s
    horizon_steps = 20     # 预测 20 步 = 10 分钟

    # 1) 读取推理样本 (1 小时输入 + m01_true 真实10分钟)
    time_hist, turbine_1h, true_times, true_speed, true_cos, true_sin = load_infer_sample(
        base_dir=base_dir,
        weather_fname="weather.csv",
        predict_fname="m01_predict.csv",
        true_fname="m01_true.csv",
        step_sec=step_sec,
        horizon_steps=horizon_steps
    )

    # 2) 构造 Time2GraphWindModel（一定要和训练 run2.py 的参数一致！！）
    seg_length = 5      # 保证 seg_length * num_segment == 1 小时时间点数
    num_segment = 4      # 24 * 5 = 120
    K = 15
    C = 300
    percentile = 10

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
        contrast_weight=0.1,
        shapelets_cache='{}/scripts/cache/{}/{}_{}_{}_{}_shapelets.cache'.format(
            module_path, farm, turbine_id,'greedy', K, seg_length
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

    # 当前"滚动窗口"的机舱数据 & 时间
    cur_turb = turbine_1h.copy()    # (>=120, 5)
    cur_times = time_hist.copy()    # (>=120,)

    # 预测结果缓存
    pred_speed_list = []
    pred_cos_list = []
    pred_sin_list = []

    weather_path = os.path.join(base_dir, "weather.csv")
    if not os.path.isfile(weather_path):
        raise FileNotFoundError(f"找不到气象数据文件: {weather_path}")

    print(f"开始自回归预测 {horizon_steps} 步...")

    prev_v = None
    for k in range(horizon_steps):
        # 取最近 L_required 个点作为输入窗口
        win_turb = cur_turb[-L_required:, :]   # (L, 5)
        win_times = cur_times[-L_required:]    # (L,)

        # 对这一段时间做气象插值 -> (L, 3) [wind_spd, cosθ, sinθ]
        weather_features = load_weather_data(weather_path, win_times)

        # 拼出模型输入: (1, L, 8)
        # 特征顺序: [功率, 温度, 风速, Cosθ机舱, Sinθ机舱, 风速气象, Cosθ气象, Sinθ气象]
        X_win = np.concatenate([win_turb, weather_features], axis=-1)  # (L, 8)
        X_win = X_win[np.newaxis, :, :]                                # (1, L, 8)

        # 单步预测：输出 [风速, Cosθ, Sinθ]，形状 (1, 3)
        y_step = m.predict(X_win)[0]   # (3,)
        v_pred = float(y_step[0])
        cos_pred = float(y_step[1])
        sin_pred = float(y_step[2])

        # === DEBUG 打印 ===
        last_speed_in_win = win_turb[-1, 2]
        # 将预测的Cos/Sin转回角度以便阅读
        pred_angle = convert_cos_sin_to_degrees(np.array([cos_pred]), np.array([sin_pred]))[0]
        last_angle = convert_cos_sin_to_degrees(np.array([win_turb[-1, 3]]), np.array([win_turb[-1, 4]]))[0]
        
        print(f"[step {k}] last_speed={last_speed_in_win:.6f}, last_dir={last_angle:.1f}°, "
              f"pred_speed={v_pred:.6f}, pred_dir={pred_angle:.1f}°")
        if prev_v is not None:
            print(f"    Δv_pred={v_pred - prev_v:.6e}")
        prev_v = v_pred
        # ==================

        pred_speed_list.append(v_pred)
        pred_cos_list.append(cos_pred)
        pred_sin_list.append(sin_pred)

        # 生成下一个时刻的时间戳（30s 后）
        new_time = cur_times[-1] + np.timedelta64(step_sec, 's')

        # 新的机舱点：功率/温度沿用上一时刻，风速和风向用预测的Cos/Sin
        last_power = cur_turb[-1, 0]
        last_temp  = cur_turb[-1, 1]
        new_point = np.array([[last_power, last_temp, v_pred, cos_pred, sin_pred]])  # (1, 5)

        # 追加到序列末尾，为下一步预测准备
        cur_turb = np.vstack([cur_turb, new_point])
        cur_times = np.append(cur_times, new_time)

    pred_speed = np.array(pred_speed_list)   # (20,)
    pred_cos   = np.array(pred_cos_list)     # (20,)
    pred_sin   = np.array(pred_sin_list)     # (20,)
    
    # 将预测的Cos/Sin转回角度
    pred_dir_deg = convert_cos_sin_to_degrees(pred_cos, pred_sin)
    # 将真实的Cos/Sin转回角度
    true_dir_deg = convert_cos_sin_to_degrees(true_cos, true_sin)

    # 5) 组织输出表格
    if len(true_times) == horizon_steps:
        time_col = true_times
    else:
        time_col = cur_times[-horizon_steps:]

    # 计算风速和风向误差
    speed_err = pred_speed - true_speed
    
    # 风向误差计算（考虑角度循环）
    dir_err_deg = []
    for pred_angle, true_angle in zip(pred_dir_deg, true_dir_deg):
        err = pred_angle - true_angle
        if err > 180:
            err -= 360
        elif err < -180:
            err += 360
        dir_err_deg.append(err)

    dir_err_deg = np.array(dir_err_deg)

    result_df = pd.DataFrame({
        "time": pd.to_datetime(time_col),
        "真实风速": true_speed,
        "预测风速": pred_speed,
        "风速误差": speed_err,
        "真实风向": true_dir_deg,
        "预测风向": pred_dir_deg,
        "风向误差": dir_err_deg,
        "预测Cos": pred_cos,
        "预测Sin": pred_sin,
        "真实Cos": true_cos,
        "真实Sin": true_sin,
    })

    out_path = os.path.join(base_dir, "m01_predict_result.csv")
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print("预测结果已保存到:", out_path)
    print("\n前5行预测结果:")
    print(result_df.head())
    
    # 打印统计信息
    print(f"\n预测统计:")
    print(f"风速 MAE: {np.abs(speed_err).mean():.4f} m/s")
    print(f"风速 RMSE: {np.sqrt((speed_err**2).mean()):.4f} m/s")
    print(f"风向 MAE: {np.abs(dir_err_deg).mean():.2f} °")
    print(f"风向 RMSE: {np.sqrt((dir_err_deg**2).mean()):.2f} °")