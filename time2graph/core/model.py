# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import torch.nn.functional as F
import random
from config import *
from time2graph.core.model_embeds import Time2GraphEmbed
from time2graph.utils.base_utils import Debugger


class ShapeletSeqRegressor(nn.Module):
    """
    使用 Transformer + mean+max pooling + 简化版 MLP。
    输入:  (B, S, D)
    输出:  (B, 3)  -> [风速, cosθ, sinθ]
    """
    def __init__(self, embed_dim, num_heads=4, num_layers=2,
                 ff_hidden_dim=256, dropout=0.1, debug=False):
        super().__init__()

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

        self.debug = debug
        self.register_buffer("debug_printed", torch.zeros(1, dtype=torch.bool))

        self.mlp = nn.Sequential(
            nn.Linear(2*embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, 3)  # 输出 [speed, cos, sin]
        )

    def encode(self, seq_emb):
        """
        返回用于对比学习的特征向量 (B, 2*D)
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

        enc = self.encoder(seq_emb)          # (B, S, D)
        mean_pool = enc.mean(dim=1)         # (B, D)
        max_pool, _ = enc.max(dim=1)        # (B, D)
        pooled = torch.cat([mean_pool, max_pool], dim=-1)   # (B, 2D)
        return pooled

    def forward(self, seq_emb, return_feat=False):
        """
        return_feat=True 时同时返回 (output, feature)
        输出:
          out: (B, 3) -> [speed, cos, sin]，其中 (cos, sin) 已经被归一化到单位圆上
        """
        pooled = self.encode(seq_emb)       # (B, 2D)
        raw_out = self.mlp(pooled)         # (B, 3)

        # 第一个分量：风速
        speed = raw_out[:, :1]             # (B, 1)

        # 后两个分量：原始方向向量 -> 归一化成单位向量
        vec = raw_out[:, 1:]               # (B, 2)
        vec_norm = torch.clamp(vec.norm(dim=-1, keepdim=True), min=1e-6)  # 防止除 0
        dir_unit = vec / vec_norm          # (B, 2) 现在 cos^2 + sin^2 ≈ 1

        out = torch.cat([speed, dir_unit], dim=-1)  # (B, 3)

        if return_feat:
            return out, pooled
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
                 data_size=8,
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

        # ===== 对比学习超参数 =====
        self.contrast_weight = kwargs.get('contrast_weight', 0.5)  # 你在 run2.py 里已经传了 0.1
        self.contrast_margin = kwargs.get('contrast_margin', 4.0)   # 可以比之前稍微大一点
        # ===== 风向损失的权重（相对于风速）=====
        self.dir_loss_weight = kwargs.get('dir_loss_weight', 4.0)

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
        print(f"Input data shape: {X.shape}")  # 打印输入数据的维度
        print(f"Embedding shape: {emb.shape}")  # 打印嵌入的形状
        print(f"Total embedding dimension: {total_dim}, D_emb: {D_emb}")
        return seq_tensor, D_emb

    def _make_negative(self, seq_batch):
        """
        构造负样本：
        seq_batch: (B, S, D)

        操作：
        - mask：随机一段时间片置为均值（而不是0，避免太假）
        - local_shuffle：只在一小段窗口内打乱顺序
        - graft：从同 batch 另一条样本嫁接一小段
        """
        B, S, D = seq_batch.shape
        neg = seq_batch.clone()

        for i in range(B):
            op = random.choice(['mask', 'local_shuffle', 'graft'])

            # 每条样本至少保留一点结构，不要改太狠
            span_ratio = random.uniform(0.2, 0.4)
            span = max(1, int(S * span_ratio))
            start = random.randint(0, S - span)

            if op == 'mask':
                # 用该样本的整体均值来“抹掉”这一段
                mean_vec = neg[i].mean(dim=0, keepdim=True)   # (1, D)
                neg[i, start:start+span, :] = mean_vec

            elif op == 'local_shuffle':
                # 只打乱一个小段内部的时间顺序
                idx = torch.randperm(span, device=neg.device)
                segment = neg[i, start:start+span, :].clone()
                neg[i, start:start+span, :] = segment[idx, :]

            elif op == 'graft' and B > 1:
                # 从 batch 里选一个别的样本，嫁接同长度的一段
                j = random.randint(0, B - 1)
                while j == i:
                    j = random.randint(0, B - 1)

                donor_start = random.randint(0, S - span)
                donor_seg = neg[j, donor_start:donor_start+span, :].clone()
                neg[i, start:start+span, :] = donor_seg

        return neg

    def _contrastive_loss(self, feat_pos, feat_neg, margin=None):
        """
        feat_pos, feat_neg: (B, F)
        希望 ||feat_pos - feat_neg|| >= margin

        loss = mean( relu( margin - dist_neg ) )
        """
        if margin is None:
            margin = self.contrast_margin

        # (B,)
        dist_neg = torch.norm(feat_pos - feat_neg, dim=-1)
        loss = F.relu(margin - dist_neg).mean()
        return loss


    # --------- 对外接口：fit & predict ---------

    def fit(self, X, Y,
            lr=1e-3,
            num_epochs=50,
            batch_size=32,
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

        # ====== 准备优化器 & 角度损失权重 ======
        optimizer = optim.Adam(self.regressor.parameters(), lr=lr)

        y_tensor = torch.from_numpy(Y).float().to(self.device)

        idx = np.arange(N)
        for epoch in range(num_epochs):
            np.random.shuffle(idx)
            epoch_reg_loss = 0.0
            epoch_contrast_loss = 0.0
            epoch_speed_loss = 0.0
            epoch_dir_loss = 0.0

            self.regressor.train()

            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch_idx = idx[start:end]
                batch_x = seq_emb[batch_idx]      # (B, S, D)
                batch_y = y_tensor[batch_idx]     # (B, 3) -> [speed, cos, sin]

                optimizer.zero_grad()

                # ------------- 正样本前向 -------------
                pred, feat_pos = self.regressor(batch_x, return_feat=True)
                # pred: (B,3) -> [speed, cos, sin]，其中 (cos, sin) 已经单位化

                # 拆开三维
                speed_pred = pred[:, 0]
                cos_pred   = pred[:, 1]
                sin_pred   = pred[:, 2]

                speed_true = batch_y[:, 0]
                cos_true   = batch_y[:, 1]
                sin_true   = batch_y[:, 2]

                # ---- 速度损失：MSE ----
                speed_loss = F.mse_loss(speed_pred, speed_true)

                # ---- 方向损失：1 - cos(Δθ)（等价于 1 - 点积）----
                dot = cos_pred * cos_true + sin_pred * sin_true
                dot = torch.clamp(dot, -1.0, 1.0)         # 数值稳定一下
                dir_loss = (1.0 - dot).mean()             # 越小方向越接近

                reg_loss = speed_loss + self.dir_loss_weight * dir_loss

                # ------------- 负样本构造 + 前向（对比学习）-------------
                neg_batch_x = self._make_negative(batch_x)      # (B, S, D)
                _, feat_neg = self.regressor(neg_batch_x, return_feat=True)

                contrast_loss = self._contrastive_loss(feat_pos, feat_neg)

                # 总 loss
                loss = reg_loss + self.contrast_weight * contrast_loss
                loss.backward()
                optimizer.step()

                bs = end - start
                epoch_reg_loss      += reg_loss.item()      * bs
                epoch_contrast_loss += contrast_loss.item() * bs
                epoch_speed_loss    += speed_loss.item()    * bs
                epoch_dir_loss      += dir_loss.item()      * bs

            epoch_reg_loss      /= N
            epoch_contrast_loss /= N
            epoch_speed_loss    /= N
            epoch_dir_loss      /= N

            if self.verbose:
                Debugger.info_print(
                    f'[Epoch {epoch+1}/{num_epochs}] '
                    f'reg_loss={epoch_reg_loss:.6f}, '
                    f'speed_loss={epoch_speed_loss:.6f}, '
                    f'dir_loss={epoch_dir_loss:.6f}, '
                    f'contrast_loss={epoch_contrast_loss:.6f}'
                )



    @torch.no_grad()
    def predict(self, X, as_numpy=True):
        """
        X: numpy, (N, L, data_size)
        返回: 预测的 [风速, cosθ, sinθ]，shape (N, 3)
        """
        assert self.regressor is not None, "model not trained yet."
        X = np.asarray(X, dtype=np.float32)
        seq_emb, _ = self._extract_shapelet_seq(X)
        self.regressor.eval()

        out = self.regressor(seq_emb)  # 可能是 Tensor，也可能是 (Tensor, feat)

        # 如果 forward 不小心返回了 (pred, feat)，这里只取 pred
        if isinstance(out, tuple):
            preds = out[0]
        else:
            preds = out

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