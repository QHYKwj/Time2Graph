# infer_all.py
# -*- coding: utf-8 -*-
"""
批量 Time2Graph 风象预测推理脚本

目录结构假定为：
    /mnt/e/Desktop/Xplanet/AI比赛/预测数据/
        m01/
            m01_predict.csv
            m01_true.csv
            weather.csv
        m02/
            m02_predict.csv
            m02_true.csv
            weather.csv
        ...
        m50/
            m50_predict.csv
            m50_true.csv
            weather.csv

模型与 shapelet 缓存目录（与训练脚本一致）：
    {module_path}/scripts/cache/风场1/m01_*.cache
    {module_path}/scripts/cache/风场2/m26_*.cache
"""

import os
import numpy as np
import pandas as pd
from config import *
from time2graph.core.model import Time2GraphWindModel
from scripts.run2 import load_weather_data


# ================== 全局配置（和训练保持一致） ==================

# 预测数据根目录（可按需修改）
PRED_ROOT = r"/mnt/e/Desktop/Xplanet/AI比赛/预测数据"

# 自回归预测相关
STEP_SEC = 30          # 每一步 30s
HORIZON_STEPS = 20     # 预测 20 步 = 10 分钟

# 模型结构参数（必须与训练 run2.py 时一致！）
SEG_LENGTH = 5
NUM_SEGMENT = 4        # 一个窗口长度 L = SEG_LENGTH * NUM_SEGMENT = 20
K = 15
C = 300
PERCENTILE = 10


# ================== 工具函数 ==================

def turbine_id_to_farm(turbine_id: str) -> str:
    """
    m01-m25 -> 风场1
    m26-m50 -> 风场2
    """
    idx = int(turbine_id[1:])  # "m01" -> 1
    if 1 <= idx <= 25:
        return "风场1"
    elif 26 <= idx <= 50:
        return "风场2"
    else:
        raise ValueError(f"非法风机编号: {turbine_id}")


def load_infer_sample_for_turbine(turbine_dir: str,
                                  turbine_id: str,
                                  step_sec: int = STEP_SEC,
                                  horizon_steps: int = HORIZON_STEPS):
    """
    对单个风机读取：
      - mXX_predict.csv: 作为历史 1h（或更长）的机舱输入
      - mXX_true.csv:    随后 10 分钟（20 步）的真实机舱数据
      - weather.csv:     对应时间段的逐小时气象数据（用于插值）

    返回：
      time_hist:   (T_in,)    历史时间戳（datetime64[s]）
      turbine_1h:  (T_in, 4)  [功率, 温度, 风速, 风向]
      true_times:  (H,)       未来 20 步时间
      true_speed:  (H,)       未来真实风速
      true_dir:    (H,)       未来真实风向
      weather_path: str       weather.csv 路径（后续插值用）
    """
    pred_fname = f"{turbine_id}_predict.csv"
    true_fname = f"{turbine_id}_true.csv"
    weather_fname = "weather.csv"

    pred_path = os.path.join(turbine_dir, pred_fname)
    true_path = os.path.join(turbine_dir, true_fname)
    weather_path = os.path.join(turbine_dir, weather_fname)

    if not os.path.isfile(pred_path):
        raise FileNotFoundError(f"缺少 {pred_fname}: {pred_path}")
    if not os.path.isfile(true_path):
        raise FileNotFoundError(f"缺少 {true_fname}: {true_path}")
    if not os.path.isfile(weather_path):
        raise FileNotFoundError(f"缺少 {weather_fname}: {weather_path}")

    # ==== 1) 历史机舱输入 ====
    df_pred = pd.read_csv(pred_path)
    df_pred["time"] = pd.to_datetime(df_pred["time"])
    df_pred = df_pred.sort_values("time").reset_index(drop=True)

    input_steps = len(df_pred)
    if input_steps == 0:
        raise ValueError(f"{pred_fname} 中没有任何数据")
    print(f"  [{turbine_id}] {pred_fname}: {input_steps} 条记录，将全部用于输入序列")

    turbine_1h = np.column_stack([
        df_pred["变频器电网侧有功功率"].values.astype(float),
        df_pred["外界温度"].values.astype(float),
        df_pred["风速"].values.astype(float),
        df_pred["风向"].values.astype(float),
    ])  # (T_in, 4)

    time_hist = df_pred["time"].astype("datetime64[s]").values  # (T_in,)

    # ==== 2) 真实未来 10 分钟 ====
    df_true = pd.read_csv(true_path)
    df_true["time"] = pd.to_datetime(df_true["time"])
    df_true = df_true.sort_values("time").reset_index(drop=True)

    total_true = len(df_true)
    if total_true < horizon_steps:
        raise ValueError(
            f"{true_fname} 只有 {total_true} 行，不足 {horizon_steps} 个 30s 点，"
            "请检查是否为完整的 10 分钟数据"
        )
    df_true = df_true.iloc[:horizon_steps].copy()

    true_speed = df_true["风速"].values.astype(float)
    true_dir = df_true["风向"].values.astype(float)
    true_times = df_true["time"].astype("datetime64[s]").values

    return time_hist, turbine_1h, true_times, true_speed, true_dir, weather_path


def infer_for_one_turbine(pred_root: str,
                          turbine_id: str,
                          step_sec: int = STEP_SEC,
                          horizon_steps: int = HORIZON_STEPS):
    """
    对单个风机执行：
      1) 参数一致性检查
      2) 加载缓存的 shapelets + regressor
      3) 自回归预测 20 步
      4) 输出 mXX_predict_true.csv
    """
    turbine_dir = os.path.join(pred_root, turbine_id)
    if not os.path.isdir(turbine_dir):
        print(f"[{turbine_id}] 目录不存在，跳过")
        return

    farm = turbine_id_to_farm(turbine_id)

    print("=" * 70)
    print(f"开始处理风机 {turbine_id} （{farm}）")

    # ----------------------------
    # 1. 加载推理样本
    # ----------------------------
    try:
        time_hist, turbine_hist, true_times, true_speed, true_dir, weather_path = \
            load_infer_sample_for_turbine(
                turbine_dir=turbine_dir,
                turbine_id=turbine_id,
                step_sec=step_sec,
                horizon_steps=horizon_steps
            )
    except Exception as e:
        print(f"[{turbine_id}] 加载数据失败：{e}")
        return

    # ----------------------------
    # 2. 参数一致性检查（K, seg_length）
    # ----------------------------
    shapelets_path = '{}/scripts/cache/{}/{}_{}_{}_{}_shapelets.cache'.format(
        module_path, farm, turbine_id, 'greedy', K, SEG_LENGTH
    )

    if not os.path.isfile(shapelets_path):
        print(f"[{turbine_id}] 缺少 shapelets_cache，跳过：{shapelets_path}")
        return

    fname = os.path.basename(shapelets_path)
    # 形如: m01_greedy_10_5_shapelets.cache
    try:
        parts = fname.split("_")
        K_cache = int(parts[2])
        seg_len_cache = int(parts[3])
    except Exception:
        print(f"[{turbine_id}] shapelets_cache 文件名格式异常，跳过：{fname}")
        return

    if K_cache != K or seg_len_cache != SEG_LENGTH:
        print(f"[{turbine_id}] 参数不一致，跳过："
              f"训练(K={K_cache}, seg={seg_len_cache}) vs 推理(K={K}, seg={SEG_LENGTH})")
        return

    print(f"  [{turbine_id}] 参数一致性检查通过")

    # ----------------------------
    # 3. 构造模型并加载 regressor
    # ----------------------------
    m = Time2GraphWindModel(
        K=K,
        C=C,
        seg_length=SEG_LENGTH,
        num_segment=NUM_SEGMENT,
        init=0,
        warp=2,
        tflag=True,
        gpu_enable=True,          # 如果想强制 CPU，可改为 False
        percentile=PERCENTILE,
        embed_mode='concate',
        batch_size=50,
        data_size=1,
        scaled=False,
        transformer_heads=4,
        transformer_layers=2,
        transformer_ff=256,
        dropout=0.1,
        verbose=True,
        shapelets_cache=shapelets_path
    )

    model_path = '{}/scripts/cache/{}/{}_embedding_t2g_model.cache'.format(
        module_path, farm, turbine_id
    )
    if not os.path.isfile(model_path):
        print(f"[{turbine_id}] 缺少 regressor 模型，跳过：{model_path}")
        return

    try:
        m.load_model(model_path)
        print(f"  [{turbine_id}] 模型已成功加载")
    except Exception as e:
        print(f"[{turbine_id}] 模型加载失败，跳过：{e}")
        return

    # ----------------------------
    # 4. 自回归循环预测 20 步
    # ----------------------------
    L_required = SEG_LENGTH * NUM_SEGMENT   # 20
    T_in, _ = turbine_hist.shape
    if T_in < L_required:
        print(f"[{turbine_id}] 输入长度 {T_in} < 窗口长度 {L_required}，跳过")
        return

    cur_turb = turbine_hist.copy()
    cur_times = time_hist.copy()

    pred_speed_list = []
    pred_dir_list = []

    print(f"  [{turbine_id}] 开始自回归预测 {horizon_steps} 步...")

    prev_v = None
    for k in range(horizon_steps):
        # 最近 L_required 个点作为输入窗口
        win_turb = cur_turb[-L_required:, :]   # (L, 4)
        win_times = cur_times[-L_required:]    # (L,)

        # 这一段时间做气象插值 -> (L, 3)
        weather_features = load_weather_data(weather_path, win_times)

        # 拼出模型输入: (1, L, 7)
        X_win = np.concatenate([win_turb, weather_features], axis=-1)  # (L, 7)
        X_win = X_win[np.newaxis, :, :]                                # (1, L, 7)

        # 单步预测: (1, 2) -> [风速, 风向]
        y_step = m.predict(X_win)[0]
        v_pred = float(y_step[0])
        dir_pred = float(y_step[1])

        last_speed_in_win = win_turb[-1, 2]
        print(f"    [step {k:02d}] last_v={last_speed_in_win:.6f}, "
              f"pred_v={v_pred:.6f}, pred_dir={dir_pred:.6f}")
        if prev_v is not None:
            print(f"             Δv_pred={v_pred - prev_v:.6e}")
        prev_v = v_pred

        pred_speed_list.append(v_pred)
        pred_dir_list.append(dir_pred)

        # 下一个时刻 30s 后的时间戳
        new_time = cur_times[-1] + np.timedelta64(step_sec, 's')

        # 新机舱点：功率/温度沿用上一时刻，风速/风向用预测
        last_power = cur_turb[-1, 0]
        last_temp = cur_turb[-1, 1]
        new_point = np.array([[last_power, last_temp, v_pred, dir_pred]])  # (1, 4)

        cur_turb = np.vstack([cur_turb, new_point])
        cur_times = np.append(cur_times, new_time)

    pred_speed = np.array(pred_speed_list)   # (H,)
    pred_dir = np.array(pred_dir_list)       # (H,)

    # ----------------------------
    # 5. 组织结果并保存
    # ----------------------------
    if len(true_times) == horizon_steps:
        time_col = true_times
    else:
        time_col = cur_times[-horizon_steps:]

    speed_err = pred_speed - true_speed
    dir_err = pred_dir - true_dir

    result_df = pd.DataFrame({
        "time": pd.to_datetime(time_col),
        "真实风速": true_speed,
        "预测风速": pred_speed,
        "风速误差": speed_err,
        "真实风向": true_dir,
        "预测风向": pred_dir,
        "风向误差": dir_err,
    })

    out_path = os.path.join(turbine_dir, f"{turbine_id}_predict_true.csv")
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")

    # 简单打印一下误差指标
    mae_v = np.mean(np.abs(speed_err))
    mae_dir = np.mean(np.abs(dir_err))
    print(f"  [{turbine_id}] 预测完成 -> 保存至: {out_path}")
    print(f"  [{turbine_id}] 风速 MAE={mae_v:.4f}, 风向 MAE={mae_dir:.4f}")


# ================== 主函数 ==================

if __name__ == "__main__":
    print("====== 批量 Time2Graph 风象预测 infer_all.py ======")
    print(f"预测数据根目录: {PRED_ROOT}")
    print(f"模型参数: K={K}, C={C}, seg_length={SEG_LENGTH}, num_segment={NUM_SEGMENT}, percentile={PERCENTILE}")
    print("====================================================")

    for i in range(1, 51):
        turbine_id = f"m{i:02d}"
        infer_for_one_turbine(PRED_ROOT, turbine_id)

    print("====== 全部风机推理结束 ======")
