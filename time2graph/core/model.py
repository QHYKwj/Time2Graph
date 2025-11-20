# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from config import *
from time2graph.core.model_embeds import Time2GraphEmbed
from time2graph.utils.base_utils import Debugger


class ShapeletSeqRegressor(nn.Module):
    """
    使用 Transformer + mean+max pooling + 简化版 MLP。
    输入:  (B, S, D)
    输出:  (B, 2)  -> 风速、风向
    """
    def __init__(self, embed_dim, num_heads=4, num_layers=2,
                 ff_hidden_dim=256, dropout=0.1, debug=False):
        super().__init__()

        # === Transformer Encoder ===
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # === debug 配置 ===
        self.debug = debug
        self.register_buffer("debug_printed", torch.zeros(1, dtype=torch.bool))

        # === 简单 MLP ===
        # 输入维度 = mean + max pooling，所以是 2 * embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, 2)   # 最后一层直接输出风速、风向
        )

    def forward(self, seq_emb):
        """
        seq_emb: (B, S, D)
        """
        B, S, D = seq_emb.shape

        # Debug — 只打印一次
        if self.debug and not bool(self.debug_printed.item()):
            with torch.no_grad():
                print(f"[DEBUG] forward: seq_emb=(B={B}, S={S}, D={D})")
                if B >= 2:
                    x0 = seq_emb[0].cpu()
                    x1 = seq_emb[1].cpu()
                    print(f"[DEBUG] sample0 mean/std = {x0.mean():.6f}/{x0.std():.6f}")
                    print(f"[DEBUG] sample1 mean/std = {x1.mean():.6f}/{x1.std():.6f}")
                    print(f"[DEBUG] L2 distance(sample0,sample1)= {torch.norm(x0 - x1):.6f}")
            self.debug_printed[...] = True

        # === Transformer 编码 ===
        enc = self.encoder(seq_emb)  # (B, S, D)

        # === mean + max pooling（特征更强）===
        mean_pool = enc.mean(dim=1)     # (B, D)
        max_pool, _ = enc.max(dim=1)    # (B, D)
        pooled = torch.cat([mean_pool, max_pool], dim=-1)   # (B, 2D)

        # === 简化 MLP ===
        out = self.mlp(pooled)   # (B, 2)
        return out



class Time2GraphWindModel(object):
    """
    用 Time2Graph 的 shapelet 嵌入 + Transformer+MLP
    做风速/风向回归的示例模型。

    - 输入 X: numpy, shape (N, L, data_size)
    - 输出 Y: numpy, shape (N, 2)  [风速, 风向] (具体含义由你定义)
    """
    def __init__(self,
                 shapelets_cache,
                 K,
                 C,
                 seg_length,
                 num_segment,
                 init=0,
                 warp=2,
                 tflag=True,
                 gpu_enable=True,
                 percentile=15,
                 embed_mode='concate',
                 batch_size=100,
                 data_size=1,
                 scaled=False,
                 device=None,
                 transformer_heads=4,
                 transformer_layers=2,
                 transformer_ff=256,
                 dropout=0.1,
                 verbose=True,
                 **kwargs):
        super().__init__()
        self.shapelets_cache = shapelets_cache
        self.K = K
        self.C = C
        self.seg_length = seg_length
        self.num_segment = num_segment
        self.init = init
        self.warp = warp
        self.tflag = tflag
        self.gpu_enable = gpu_enable
        self.percentile = percentile
        self.embed_mode = embed_mode  # 必须为 'concate'
        self.batch_size = batch_size
        self.data_size = data_size
        self.scaled = scaled
        self.verbose = verbose
        self.transformer_heads = transformer_heads
        self.transformer_layers = transformer_layers
        self.transformer_ff = transformer_ff
        self.dropout = dropout

        self.device = device or (
            'cuda' if torch.cuda.is_available() and gpu_enable else 'cpu'
        )

        # Time2GraphEmbed 用来学习 shapelet + 图嵌入
        self.t2g = Time2GraphEmbed(
            kernel=None,    # 不再用外部分类器
            K=K,
            C=C,
            seg_length=seg_length,
            opt_metric='accuracy',  # 占位，不重要
            warp=warp,
            tflag=tflag,
            gpu_enable=gpu_enable,
            percentile=percentile,
            mode=embed_mode,
            batch_size=batch_size,
            **kwargs
        )

        # 如果有缓存的 shapelet，加载
        if os.path.isfile(self.shapelets_cache):
            self.t2g.load_shapelets(fpath=self.shapelets_cache)
            Debugger.info_print(
                f'load shapelets from cache {self.shapelets_cache}...'
            )

        # 回归器先占位，等知道 embedding 维度后再构建
        self.regressor = None

    # --------- 内部工具函数 ---------

    def _learn_shapelets_and_embedding(self, X, Y=None, cache_dir='./cache'):
        """
        学 shapelet + 图嵌入 (DeepWalk)，只做一次。
        """
         # -------- 1) 先保证有个合法的 y 给 shapelet 学习 --------
        if Y is None:
            # 无监督：随便给一类，长度和 X 样本数一致
            y_for_shapelet = np.zeros(X.shape[0], dtype=int)
        else:
            # 有监督：如果你以后想用分类标签，可以在这里处理
            Y_arr = np.asarray(Y)
            if Y_arr.ndim == 2:
                # 比如回归标签 (N,2)，我们随便用第一个维度离散化一下
                # 这里简单一点，直接全 0 也可以
                y_for_shapelet = np.zeros(Y_arr.shape[0], dtype=int)
            else:
                y_for_shapelet = Y_arr.astype(int)
                
        if self.t2g.shapelets is None:
            Debugger.info_print('learning shapelets...')
            self.t2g.learn_shapelets(
                x=X,
                y=y_for_shapelet,
                num_segment=self.num_segment,
                data_size=self.data_size,
                num_batch=int(X.shape[0] // self.batch_size)
            )
            self.t2g.save_shapelets(fpath=self.shapelets_cache)
            Debugger.info_print(
                f'saving shapelets cache to {self.shapelets_cache}'
            )

        if self.t2g.sembeds is None:
            Debugger.info_print('training embedding model (DeepWalk)...')
            self.t2g.fit_embedding_model(
                x=X,
                y=y_for_shapelet,
                cache_dir=cache_dir
            )

    def _extract_shapelet_seq(self, X):
        """
        调用 Time2GraphEmbed.embed 得到时间序列 embedding，
        假设 embed_mode='concate'，返回 (N, num_segment * D_emb)，
        然后 reshape 成 (N, num_segment, D_emb)，作为“shapelet 序列”。

        返回: seq_emb: torch.Tensor, shape (N, S, D)
        """
        # t2g.embed 返回 numpy
        emb = self.t2g.embed(x=X, init=self.init)  # (N, S * D_emb)
        N, total_dim = emb.shape
        assert total_dim % self.num_segment == 0, \
            "total embedding dim must be divisible by num_segment"
        D_emb = total_dim // self.num_segment
        seq = emb.reshape(N, self.num_segment, D_emb)  # (N, S, D)
        
         # 简单 debug：看下前两个样本的差异
        if N >= 2:
            diff = np.linalg.norm(seq[0] - seq[1])
            Debugger.info_print(f"[DEBUG] seq_emb[0] vs seq_emb[1] L2 = {diff:.6f}")
        
        seq_tensor = torch.from_numpy(seq).float().to(self.device)
        return seq_tensor, D_emb

    # --------- 对外接口：fit & predict ---------

    def fit(self, X, Y,
            lr=1e-3,
            num_epochs=50,
            batch_size=64,
            cache_dir='./cache'):
        """
        训练过程：
        1) 用 Time2Graph 学习 shapelet + 图嵌入
        2) 把每条序列转成 shapelet 序列 (N, S, D)
        3) 用 Transformer+MLP 回归 Y (N, 2)

        X: numpy, (N, L, data_size)
        Y: numpy, (N, 2)
        """
        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)
        N = X.shape[0]

        # 1) 学 shapelet + 图嵌入
        self._learn_shapelets_and_embedding(X, Y=None, cache_dir=cache_dir)

        # 2) 提取 shapelet 序列
        seq_emb, D_emb = self._extract_shapelet_seq(X)
        self.embed_dim = D_emb
        # 3) 构建回归器
        if self.regressor is None:
            self.regressor = ShapeletSeqRegressor(
                embed_dim=D_emb,
                num_heads=self.transformer_heads,
                num_layers=self.transformer_layers,
                ff_hidden_dim=self.transformer_ff,
                dropout=self.dropout,
                debug=True   # 先开着调试
            ).to(self.device)

        # 准备优化器和损失
        optimizer = optim.Adam(self.regressor.parameters(), lr=lr)
        criterion = nn.MSELoss()  # 基本版：MSE；风向你可以改成角度损失

        y_tensor = torch.from_numpy(Y).float().to(self.device)

        # 简单 mini-batch 训练
        idx = np.arange(N)
        for epoch in range(num_epochs):
            np.random.shuffle(idx)
            epoch_loss = 0.0
            self.regressor.train()
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch_idx = idx[start:end]
                batch_x = seq_emb[batch_idx]      # (B, S, D)
                batch_y = y_tensor[batch_idx]     # (B, 2)

                optimizer.zero_grad()
                pred = self.regressor(batch_x)    # (B, 2)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * (end - start)

            epoch_loss /= N
            if self.verbose:
                Debugger.info_print(
                    f'[Epoch {epoch+1}/{num_epochs}] train_loss={epoch_loss:.6f}'
                )

    @torch.no_grad()
    def predict(self, X, as_numpy=True):
        """
        X: numpy, (N, L, data_size)
        返回: 预测的 [风速, 风向]，shape (N, 2)
        """
        assert self.regressor is not None, "model not trained yet."
        X = np.asarray(X, dtype=np.float32)
        seq_emb, _ = self._extract_shapelet_seq(X)
        self.regressor.eval()
        preds = self.regressor(seq_emb)  # (N, 2)
        if as_numpy:
            return preds.cpu().numpy()
        return preds



    def save_model(self, fpath, **kwargs):
        """
        保存 Time2GraphWindModel：
        - 用 pickle 保存除 regressor 以外的所有属性
        - 用 torch.save 单独保存 regressor.state_dict()
        """
        state = self.__dict__.copy()
        regressor = state.pop('regressor', None)  # 拿出来，避免 pickle 它（里面有 lambda）

        # 1) 保存其它内容
        with open(fpath, 'wb') as f:
            pickle.dump(state, f)

        # 2) 单独保存 regressor 权重
        if regressor is not None:
            reg_path = fpath + ".pt"
            torch.save(regressor.state_dict(), reg_path)
            Debugger.info_print(f"regressor state_dict saved to {reg_path}")

    def load_model(self, fpath, **kwargs):
        """
        从文件恢复：
        - 先用 pickle 读回除 regressor 外的属性
        - 根据保存的超参数重建一个 ShapeletSeqRegressor
        - 再从 .pt 文件里加载 state_dict
        """
        # 1) 先恢复除 regressor 外的状态
        with open(fpath, 'rb') as f:
            paras = pickle.load(f)
        for key, val in paras.items():
            self.__dict__[key] = val

        # 2) 如果有 regressor 的权重文件，就重建 + 加载
        reg_path = fpath + ".pt"
        if os.path.exists(reg_path):
            # 这里假设在 fit 的时候已经存了 self.embed_dim / transformer_heads 等
            self.regressor = ShapeletSeqRegressor(
                embed_dim=self.embed_dim,
                num_heads=self.transformer_heads,
                num_layers=self.transformer_layers,
                ff_hidden_dim=self.transformer_ff,
                dropout=self.dropout
            ).to(self.device)
            self.regressor.load_state_dict(
                torch.load(reg_path, map_location=self.device)
            )
            Debugger.info_print(f"regressor restored from {reg_path}")
        else:
            Debugger.info_print("no regressor state_dict found, need to retrain regressor")