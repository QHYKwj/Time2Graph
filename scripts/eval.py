# scripts/eval_m01.py
# -*- coding: utf-8 -*-
import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from config import *
from time2graph.core.model import Time2GraphWindModel

# 复用 run2.py 里已经写好的函数（不会触发训练，因为有 if __name__ == "__main__"）
from scripts.run2 import (
    load_full_dataset,
    build_t2g_dataset,
    filter_invalid_samples,
)

if __name__ == "__main__":
    # -------- 跟训练时保持一致的配置 --------
    base_dir = '/mnt/e/Desktop/Xplanet/AI比赛/训练集'
    farm = '风场1'
    turbine_id = 'm01'
    train_end_date = "2019-06-30"

    # 你训练时用的参数：
    # python scripts/run2.py --K 10 --C 300 --num_segment 4 --seg_length 10 \
    #   --embed concate --transformer_heads 4 --transformer_layers 2 \
    #   --lr 0.001 --percentile 5 --farm 风场1 --turbine_id m01 --num_epochs 10
    K = 10
    C = 300
    seg_length = 10
    num_segment = 4
    percentile = 5

    # -------- 1) 用和训练时完全一样的方式，构造 X_test, Y_test --------
    # 这里不会训练，只是做预处理
    x_train_seq, y_train_seq, x_test_seq, y_test_seq = load_full_dataset(
        base_dir, farm, turbine_id, train_end_date
    )
    X_test, Y_test = build_t2g_dataset(
        x_test_seq, y_test_seq,
        seg_length=seg_length,
        num_segment=num_segment
    )
    X_test, Y_test = filter_invalid_samples(X_test, Y_test)
    print(f"[EVAL] {farm}/{turbine_id} -> X_test: {X_test.shape}, Y_test: {Y_test.shape}")

    # -------- 2) 构造一个“壳子模型”，参数要跟训练时一致 --------
    m = Time2GraphWindModel(
        K=K,
        C=C,
        seg_length=seg_length,
        num_segment=num_segment,
        init=0,
        warp=2,
        tflag=True,
        gpu_enable=False,          # 评估阶段也建议先关 GPU，省事
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
        shapelets_cache='{}/scripts/cache/{}/{}_{}_{}_{}_shapelets.cache'.format(
            module_path, farm, turbine_id, 'greedy', K, seg_length
        )
    )

    # -------- 3) 从缓存中加载已经训练好的模型 --------
    model_path = '{}/scripts/cache/{}/{}_embedding_t2g_model.cache'.format(
        module_path, farm, turbine_id
    )
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"找不到缓存模型: {model_path}")

    print("[EVAL] loading cached model from:", model_path)
    m.load_model(model_path)   # 会自动加载 .cache 和 .pt 里的 regressor

    # -------- 4) 分批预测，避免一次性 OOM --------
    print("[EVAL] start batched prediction...")
    batch_size = 256   # 可以根据内存改小一点，比如 128
    N = X_test.shape[0]
    all_preds = []

    for i in range(0, N, batch_size):
        x_batch = X_test[i:i + batch_size]   # (B, L, 7)
        preds = m.predict(x_batch)           # (B, 2)
        all_preds.append(preds)

    y_pred = np.vstack(all_preds)            # (N, 2)
    print("[EVAL] prediction done, y_pred shape:", y_pred.shape)

    # -------- 5) 计算指标：风速 RMSE / MAE，风向 MAE --------
    wind_speed_pred = y_pred[:, 0]
    wind_speed_true = Y_test[:, 0]
    wind_dir_pred   = y_pred[:, 1]
    wind_dir_true   = Y_test[:, 1]

    mse_speed = mean_squared_error(wind_speed_true, wind_speed_pred)
    rmse_speed = np.sqrt(mse_speed)
    mae_speed = mean_absolute_error(wind_speed_true, wind_speed_pred)
    mae_dir   = mean_absolute_error(wind_dir_true, wind_dir_pred)

    print(f"[EVAL RESULT] {farm}/{turbine_id}")
    print(f"  风速 RMSE: {rmse_speed:.4f}")
    print(f"  风速 MSE : {mse_speed:.4f}")
    print(f"  风速 MAE : {mae_speed:.4f}")
    print(f"  风向 MAE : {mae_dir:.4f}")
