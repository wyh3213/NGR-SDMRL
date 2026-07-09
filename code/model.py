from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Normal

from layers import (
    GCN2Conv,
    EncoderLayer,
    InnerProductDecoder,
    SpikingEncoder,
    WeakDecoder,
    init_params,
)


class GaussianPolicyHead(nn.Module):
    """
    Gaussian policy head used by PPO-style MORL.
    """

    def __init__(
            self,
            input_dim: int = 256,
            latent_dim: int = 256,
            log_std_min: float = -5.0,
            log_std_max: float = 0.5,
            min_std: float = 1e-4,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.mean_head = nn.Linear(input_dim, latent_dim)
        self.log_std_head = nn.Linear(input_dim, latent_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.min_std = min_std

    def _encode_stats(self, pair: Tensor, std_scale: float = 1.0) -> Tuple[Tensor, Tensor]:
        pair = self.norm(pair)
        mean = self.mean_head(pair)
        log_std = self.log_std_head(pair).clamp(self.log_std_min, self.log_std_max)
        std = (torch.exp(log_std) * std_scale).clamp_min(self.min_std)
        return mean, std

    def forward(
            self,
            pair: Tensor,
            latent_action: Optional[Tensor] = None,
            return_rl: bool = False,
            std_scale: float = 1.0,
    ):
        mean, std = self._encode_stats(pair, std_scale=std_scale)
        dist = Normal(mean, std)
        if latent_action is None:
            latent_action = dist.rsample()

        latent_dim = max(int(latent_action.shape[-1]), 1)
        latent_log_prob = dist.log_prob(latent_action).sum(dim=-1) / float(latent_dim)
        latent_entropy = dist.entropy().sum(dim=-1) / float(latent_dim)

        if return_rl:
            rl_info = {
                "latent_action": latent_action,
                "latent_mean": mean,
                "latent_std": std,
                "latent_log_prob": latent_log_prob,
                "latent_entropy": latent_entropy,
            }
            return latent_action, mean, std, rl_info

        return latent_action, mean, std

    def sample_mean(self, pair: Tensor, std_scale: float = 1.0):
        mean, std = self._encode_stats(pair, std_scale=std_scale)
        return mean, mean, std


class DyReLU(nn.Module):
    """
    Learnable affine ReLU that replaces the original tanh-based DyT block.
    """

    def __init__(self, channels: int, init_alpha: float = 1.0):
        super().__init__()
        self.alpha = nn.Parameter(torch.full((1,), float(init_alpha)))
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.alpha * x)
        return x * self.gamma + self.beta


class GANIB(nn.Module):
    """
    GANIB model with PPO/MORL integration + Spiking Neural Network + Self-Distillation.

    Backbone:
        - Residual Graph Convolution (GCN-II) for disease / metabolite similarity networks
        - Spiking Encoder: LIF 神经元对图特征进行脉冲时序编码 (新增)
        - Graph Transformer encoder with neighbor attention aggregation
    Self-Distillation (新增):
        - Temporal Self-Distillation (TSD): T_t 步脉冲输出指导 T_s 步输出
        - Spatial Self-Distillation (SSD): 最终预测指导弱解码器中间预测
    Policy:
        - Gaussian policy head for latent-space exploration (PPO)
    Decoder:
        - Dual-MLP gating + inner-product decoder
    """

    def __init__(
            self,
            hops,
            output_dim,
            input_dim,
            pe_dim,
            num_dis,
            num_meta,
            graphformer_layers,
            num_heads,
            hidden_dim,
            ffn_dim,
            dropout_rate,
            GCNII_layers,
            graph_input_dim: Optional[int] = None,
            policy_dim: int = 256,
            decoder_hidden_dim: int = 128,
            # ---- 新增: SNN 参数 ----
            snn_T_s: int = 4,
            snn_T_t: int = 8,
            snn_tau: float = 2.0,
            snn_threshold: float = 1.0,
            # ---- 新增: 自蒸馏参数 ----
            enable_distillation: bool = True,
            weak_decoder_hidden: int = 128,
    ):
        super().__init__()
        self.seq_len = hops + 1
        self.pe_dim = pe_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.graphformer_layers = graphformer_layers
        self.dropout_rate = dropout_rate
        self.num_dis = num_dis
        self.num_meta = num_meta
        self.GCNII_layers = GCNII_layers

        self.graph_hidden_dim = max(1, int(self.hidden_dim // 2))
        self.graph_input_dim = self.graph_hidden_dim if graph_input_dim is None else int(graph_input_dim)
        self.fused_dim = self.graph_hidden_dim + self.hidden_dim
        self.policy_dim = max(int(policy_dim), self.fused_dim)
        self.decoder_hidden_dim = max(int(decoder_hidden_dim), self.output_dim)

        # ---- SNN 时间步参数 ----
        self.snn_T_s = snn_T_s
        self.snn_T_t = snn_T_t
        self.enable_distillation = enable_distillation

        # ---- Graph input projection ----
        if self.graph_input_dim == self.graph_hidden_dim:
            self.graph_input_proj = nn.Identity()
        else:
            self.graph_input_proj = nn.Linear(self.graph_input_dim, self.graph_hidden_dim)

        # ---- Residual Graph Convolution (GCN-II) ----
        self.convs = nn.ModuleList([
            GCN2Conv(channels=self.graph_hidden_dim, alpha=0.1, theta=1, layer=i + 1)
            for i in range(self.GCNII_layers)
        ])

        # ---- 新增: 脉冲编码器 (Spiking Encoder) ----
        self.spiking_encoder = SpikingEncoder(
            dim=self.graph_hidden_dim,
            tau=snn_tau,
            threshold=snn_threshold,
            dropout=self.dropout_rate,
        )

        # ---- Graph Transformer Encoder ----
        self.att_embeddings_nope = nn.Linear(self.input_dim, self.hidden_dim)
        self.layers = nn.ModuleList([
            EncoderLayer(self.hidden_dim, self.ffn_dim, self.dropout_rate, self.num_heads)
            for _ in range(self.graphformer_layers)
        ])
        self.final_ln = nn.LayerNorm(self.hidden_dim)
        self.attn_layer = nn.Linear(2 * self.hidden_dim, 1)

        # ---- Dual-MLP gating decoder ----
        self.g1a = DyReLU(self.policy_dim, 1.0)
        self.g1b = DyReLU(self.policy_dim, 1.0)

        self.mlp = nn.Sequential(
            nn.Linear(self.fused_dim, self.policy_dim),
            self.g1a,
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.policy_dim, self.decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.decoder_hidden_dim, self.output_dim),
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(self.fused_dim, self.policy_dim),
            self.g1b,
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.policy_dim, self.decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.decoder_hidden_dim, self.output_dim),
        )

        self.decoder = InnerProductDecoder(self.output_dim, self.dropout_rate, self.num_dis)

        # ---- Policy head (PPO/MORL) ----
        self.policy_proj = nn.Linear(self.fused_dim, self.policy_dim)
        self.policy_head = GaussianPolicyHead(input_dim=self.policy_dim, latent_dim=self.policy_dim)
        self.residual_proj = nn.Linear(self.policy_dim, self.fused_dim)

        # ---- 新增: 弱解码器 (Spatial Self-Distillation) ----
        if self.enable_distillation:
            self.weak_decoder = WeakDecoder(
                input_dim=self.fused_dim,
                hidden_dim=max(weak_decoder_hidden, self.output_dim),
                output_dim=self.output_dim,
                num_dis=self.num_dis,
                dropout=self.dropout_rate,
            )
        else:
            self.weak_decoder = None

        self.apply(lambda module: init_params(module, n_layers=self.graphformer_layers))

    # ------------------------------------------------------------------
    # Backbone encoding
    # ------------------------------------------------------------------
    def _gcnii_forward(self, dis_data, meta_data) -> Tensor:
        """
        纯 GCN-II 前向传播 (不含脉冲编码)
        返回拼接后的疾病+代谢物图特征, shape (num_dis + num_meta, graph_hidden_dim)
        """
        x_0_dis = self.graph_input_proj(dis_data.x)
        x_dis = x_0_dis
        for conv in self.convs:
            x_dis = conv(x_dis, x_0_dis, dis_data.edge_index)

        x_0_meta = self.graph_input_proj(meta_data.x)
        x_meta = x_0_meta
        for conv in self.convs:
            x_meta = conv(x_meta, x_0_meta, meta_data.edge_index)

        return torch.cat((x_dis, x_meta), dim=0)

    def _encode_graph_branch(self, dis_data, meta_data, T: Optional[int] = None) -> Tensor:
        """
        图分支编码: GCN-II → SpikingEncoder

        Args:
            T: 脉冲模拟时间步, 默认使用 T_s
        Returns:
            脉冲编码后的图特征, shape (num_dis + num_meta, graph_hidden_dim)
        """
        if T is None:
            T = self.snn_T_s
        graph_raw = self._gcnii_forward(dis_data, meta_data)
        return self.spiking_encoder(graph_raw, T=T)

    def _encode_transformer_branch(self, processed_features: Tensor) -> Tensor:
        tensor = self.att_embeddings_nope(processed_features)
        for enc_layer in self.layers:
            tensor = enc_layer(tensor)
        x_former = self.final_ln(tensor)

        target = x_former[:, 0, :].unsqueeze(1).repeat(1, self.seq_len - 1, 1)
        node_tensor, neighbor_tensor = torch.split(x_former, [1, self.seq_len - 1], dim=1)
        layer_atten = self.attn_layer(torch.cat((target, neighbor_tensor), dim=2))
        layer_atten = F.softmax(layer_atten, dim=1)
        neighbor_tensor = torch.sum(neighbor_tensor * layer_atten, dim=1, keepdim=True)
        return (node_tensor + neighbor_tensor).squeeze(1)

    def encode_backbone(self, processed_features, dis_data, meta_data) -> Tensor:
        """
        骨干编码 (MORL 和推理使用, 默认 T_s 时间步)
        返回 shape (num_dis + num_meta, fused_dim)
        """
        x_gcnii = self._encode_graph_branch(dis_data, meta_data, T=self.snn_T_s)
        x_former = self._encode_transformer_branch(processed_features)
        return torch.cat((x_gcnii, x_former), dim=1)

    # ------------------------------------------------------------------
    # 新增: 自蒸馏前向传播
    # ------------------------------------------------------------------
    def distillation_forward(
            self,
            processed_features,
            dis_data,
            meta_data,
            T_student: Optional[int] = None,
            T_teacher: Optional[int] = None,
    ) -> dict:
        """
        自蒸馏前向传播 — 计算 TSD 和 SSD 所需的所有输出

        设计要点:
        1. GCN-II 只前向传播一次, SpikingEncoder 运行两次 (T_s 和 T_t)
        2. 教师信号全部 detach, 梯度不回传到 Policy Head
        3. 学生信号的梯度只更新骨干网络 + SpikingEncoder + WeakDecoder

        Returns:
            dict with keys:
                - graph_student: T_s 步脉冲图特征 (有梯度)
                - graph_teacher: T_t 步脉冲图特征 (无梯度)
                - y_weak: 弱解码器预测 (有梯度)
                - y_teacher: 完整模型确定性预测 (无梯度)
        """
        if T_student is None:
            T_student = self.snn_T_s
        if T_teacher is None:
            T_teacher = self.snn_T_t

        # 1. GCN-II 前向 (计算一次, 共享给学生和教师)
        graph_raw = self._gcnii_forward(dis_data, meta_data)

        # 2. 学生脉冲编码 (T_student 步, 保留梯度)
        graph_student = self.spiking_encoder(graph_raw, T=T_student)

        # 3. 教师脉冲编码 (T_teacher 步, 无梯度)
        with torch.no_grad():
            graph_teacher = self.spiking_encoder(graph_raw, T=T_teacher)

        # 4. 组装学生骨干输出
        x_former = self._encode_transformer_branch(processed_features)
        residual_student = torch.cat((graph_student, x_former), dim=1)

        # 5. 弱解码器预测 (SSD 学生, 保留梯度)
        y_weak = None
        if self.weak_decoder is not None:
            y_weak = self.weak_decoder(residual_student)

        # 6. 完整模型确定性预测 (SSD 教师, 无梯度)
        with torch.no_grad():
            policy_input = self.build_policy_input(residual_student)
            latent, _, _ = self.policy_head.sample_mean(policy_input)
            y_teacher, _, _ = self.decode_from_latent(latent, residual_student)

        return {
            "graph_student": graph_student,
            "graph_teacher": graph_teacher,
            "y_weak": y_weak,
            "y_teacher": y_teacher,
            "residual_student": residual_student,
        }

    def compute_distillation_loss(
            self,
            processed_features,
            dis_data,
            meta_data,
            alpha_tsd: float = 1.0,
            beta_ssd: float = 1.0,
            T_student: Optional[int] = None,
            T_teacher: Optional[int] = None,
    ) -> Tuple[Tensor, dict]:
        """
        计算自蒸馏损失

        L_distill = α * L_tsd + β * L_ssd

        其中:
            L_tsd = ||graph_student - sg(graph_teacher)||²   (时间自蒸馏)
            L_ssd = ||y_weak - sg(y_teacher)||²               (空间自蒸馏)

        sg(·) 表示 stop-gradient, 确保梯度不流过教师路径

        Returns:
            total_distill_loss: 标量损失
            stats: 各项损失的详细数值
        """
        outputs = self.distillation_forward(
            processed_features, dis_data, meta_data,
            T_student=T_student, T_teacher=T_teacher,
        )

        # TSD: 时间自蒸馏损失
        loss_tsd = F.mse_loss(outputs["graph_student"], outputs["graph_teacher"].detach())

        # SSD: 空间自蒸馏损失
        loss_ssd = torch.tensor(0.0, device=outputs["graph_student"].device)
        if outputs["y_weak"] is not None and outputs["y_teacher"] is not None:
            loss_ssd = F.mse_loss(outputs["y_weak"], outputs["y_teacher"].detach())

        total_loss = alpha_tsd * loss_tsd + beta_ssd * loss_ssd

        stats = {
            "loss_tsd": float(loss_tsd.item()),
            "loss_ssd": float(loss_ssd.item()),
            "loss_distill_total": float(total_loss.item()),
        }
        return total_loss, stats

    # ------------------------------------------------------------------
    # Policy interface (for MORL trainer) — 保持不变
    # ------------------------------------------------------------------
    def build_policy_input(self, residual: Tensor) -> Tensor:
        return self.policy_proj(residual)

    def act(
            self,
            policy_input: Tensor,
            latent_action: Optional[Tensor] = None,
            return_rl: bool = False,
            rl_std_scale: float = 1.0,
    ):
        return self.policy_head(
            policy_input,
            latent_action=latent_action,
            return_rl=return_rl,
            std_scale=rl_std_scale,
        )

    def decode_from_latent(self, latent_action: Tensor, residual: Tensor):
        latent_dim = max(int(latent_action.shape[-1]), 1)
        action_temperature = max(math.sqrt(latent_dim) / 4.0, 1.0)
        hidden = F.softmax(latent_action / action_temperature, dim=-1)
        hidden = self.residual_proj(hidden) + residual

        embeddings1 = self.mlp(hidden)
        embeddings2 = self.mlp2(hidden)
        embeddings = embeddings1 * embeddings2
        x1 = self.decoder(embeddings)
        return x1, hidden, embeddings

    # ------------------------------------------------------------------
    # Forward variants — 保持不变
    # ------------------------------------------------------------------
    def forward_from_encoded(
            self,
            residual: Tensor,
            policy_input: Optional[Tensor] = None,
            latent_action: Optional[Tensor] = None,
            return_rl: bool = False,
            rl_std_scale: float = 1.0,
    ):
        if policy_input is None:
            policy_input = self.build_policy_input(residual)

        if return_rl or latent_action is not None:
            latent_action, vec_mean, vec_std, rl_info = self.act(
                policy_input,
                latent_action=latent_action,
                return_rl=True,
                rl_std_scale=rl_std_scale,
            )
            x1, hidden, embeddings = self.decode_from_latent(latent_action, residual)
            rl_info.update({
                "policy_input": policy_input,
                "hidden_after_policy": hidden,
                "embeddings": embeddings,
            })
            return x1, vec_mean, vec_std, rl_info

        latent_action, vec_mean, vec_std = self.act(
            policy_input,
            latent_action=None,
            return_rl=False,
            rl_std_scale=rl_std_scale,
        )
        x1, _, _ = self.decode_from_latent(latent_action, residual)
        return x1, vec_mean, vec_std

    def forward(
            self,
            processed_features,
            dis_data,
            meta_data,
            latent_action: Optional[Tensor] = None,
            return_rl: bool = False,
            rl_std_scale: float = 1.0,
    ):
        residual = self.encode_backbone(processed_features, dis_data, meta_data)
        policy_input = self.build_policy_input(residual)
        return self.forward_from_encoded(
            residual=residual,
            policy_input=policy_input,
            latent_action=latent_action,
            return_rl=return_rl,
            rl_std_scale=rl_std_scale,
        )

    # ------------------------------------------------------------------
    # Deterministic inference — 保持不变
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_from_encoded(
            self,
            residual: Tensor,
            policy_input: Optional[Tensor] = None,
            rl_std_scale: float = 1.0,
    ):
        if policy_input is None:
            policy_input = self.build_policy_input(residual)
        latent_action, vec_mean, vec_std = self.policy_head.sample_mean(
            policy_input,
            std_scale=rl_std_scale,
        )
        x1, _, _ = self.decode_from_latent(latent_action, residual)
        return x1, vec_mean, vec_std

    @torch.no_grad()
    def predict_deterministic(self, processed_features, dis_data, meta_data, rl_std_scale: float = 1.0):
        residual = self.encode_backbone(processed_features, dis_data, meta_data)
        policy_input = self.build_policy_input(residual)
        latent_action, vec_mean, vec_std = self.policy_head.sample_mean(
            policy_input,
            std_scale=rl_std_scale,
        )
        x1, _, _ = self.decode_from_latent(latent_action, residual)
        return x1, vec_mean, vec_std

