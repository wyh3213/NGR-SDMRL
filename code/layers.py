import math
from torch import nn
import torch.nn.functional as F
from math import log
from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.typing import Adj, OptTensor
from utils import glorot

from timm.models.vision_transformer import _cfg, Mlp
from timm.models.layers import DropPath
from einops import rearrange

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ====================================================================
# Residual Graph Convolutional Coding Layer (原有)
# ====================================================================
class GCN2Conv(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, channels: int, alpha: float, theta: float = None,
                 layer: int = None, shared_weights: bool = True,
                 cached: bool = False, add_self_loops: bool = True,
                 normalize: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.channels = channels
        self.alpha = alpha
        self.beta = 1.
        if theta is not None or layer is not None:
            assert theta is not None and layer is not None
            self.beta = log(theta / layer + 1)
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight1 = Parameter(torch.Tensor(channels, channels))

        if shared_weights:
            self.register_parameter('weight2', None)
        else:
            self.weight2 = Parameter(torch.Tensor(channels, channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight1)
        glorot(self.weight2)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, x_0: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, self.flow, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

        x.mul_(1 - self.alpha)
        x_0 = self.alpha * x_0[:x.size(0)]

        if self.weight2 is None:
            out = x.add_(x_0)
            out = torch.addmm(out, out, self.weight1, beta=1. - self.beta,
                              alpha=self.beta)
        else:
            out = torch.addmm(x, x, self.weight1, beta=1. - self.beta,
                              alpha=self.beta)
            out = out + torch.addmm(x_0, x_0, self.weight2,
                                    beta=1. - self.beta, alpha=self.beta)

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'alpha={self.alpha}, beta={self.beta})')


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


def gelu(x):
    """
    GELU activation
    https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))


class AttentionTSSA(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        d = num_heads
        self.heads = num_heads

        self.attend = nn.Softmax(dim=1)
        self.attn_drop = nn.Dropout(attn_drop)

        self.qkv = nn.Linear(dim, dim, bias=qkv_bias)

        self.temp = nn.Parameter(torch.ones(num_heads, 1))

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_drop)
        )

    def forward(self, x):
        w = rearrange(self.qkv(x), 'b n (h d) -> b h n d', h=self.heads)

        b, h, N, d = w.shape

        w_normed = torch.nn.functional.normalize(w, dim=-2)
        w_sq = w_normed ** 2

        # Pi from Eq. 10 in the paper
        Pi = self.attend(torch.sum(w_sq, dim=-1) * self.temp)  # b * h * n

        dots = torch.matmul((Pi / (Pi.sum(dim=-1, keepdim=True) + 1e-8)).unsqueeze(-2), w ** 2)
        attn = 1. / (1 + dots)
        attn = self.attn_drop(attn)

        out = - torch.mul(w.mul(Pi.unsqueeze(-1)), attn)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# Feedforward Network (FFN)
class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


# Multi-head Self-Attention (MSA)
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5
        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)
        self.output_layer = nn.Linear(num_heads * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()
        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k)
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k)
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        k = k.transpose(1, 2).transpose(2, 3)

        q = q * self.scale
        x = torch.matmul(q, k)
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)

        x = self.att_dropout(x)
        x = x.matmul(v)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, -1, self.num_heads * d_v)
        x = self.output_layer(x)
        assert x.size() == orig_q_size
        return x


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, num_heads):
        super(EncoderLayer, self).__init__()
        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attentionTSSA = AttentionTSSA(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=False,
            attn_drop=dropout_rate,
            proj_drop=dropout_rate
        )
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None):
        y = self.self_attention_norm(x)
        y = self.self_attentionTSSA(y)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate, num_heads):
        super(DecoderLayer, self).__init__()

        self.self_attention_norm1 = nn.LayerNorm(hidden_size)
        self.self_attention1 = MultiHeadAttention(
            hidden_size, dropout_rate, num_heads)
        self.self_attention_dropout1 = nn.Dropout(dropout_rate)

        self.self_attention_norm2 = nn.LayerNorm(hidden_size)
        self.self_attention2 = MultiHeadAttention(
            hidden_size, dropout_rate, num_heads)
        self.self_attention_dropout2 = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, z, attn_bias=None):
        y = self.self_attention_norm1(x)
        y = self.self_attention1(y, y, y, attn_bias)
        y = self.self_attention_dropout1(y)
        x = x + y

        y = self.self_attention_norm2(x)
        y = self.self_attention2(y, z, z, attn_bias)
        y = self.self_attention_dropout2(y)

        x = z + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class InnerProductDecoder(nn.Module):
    """
    inner product decoder
    """

    def __init__(self, output_node_dim, dropout, num_dis):
        super(InnerProductDecoder, self).__init__()
        self.output_node_dim = output_node_dim
        self.dropout = dropout
        self.num_dis = num_dis
        self.weight = nn.Parameter(torch.empty(size=(self.output_node_dim, self.output_node_dim)))
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)

    def forward(self, inputs):
        inputs = F.dropout(inputs, self.dropout)
        Dis = inputs[0:self.num_dis, :]
        Meta = inputs[self.num_dis:, :]
        Meta = torch.mm(Meta, self.weight)
        Dis = torch.t(Dis)
        x = torch.mm(Meta, Dis)
        outputs = torch.sigmoid(x)
        return outputs


# ====================================================================
# 新增: 脉冲神经网络 (SNN) 组件
# 参考: TSSD (Temporal-Spatial Self-Distillation) 论文
# ====================================================================

class SurrogateSpike(torch.autograd.Function):
    """
    矩形代理梯度函数 (Rectangular Surrogate Gradient)

    前向传播: 阶跃函数 (膜电位 >= 阈值 → 发放脉冲)
    反向传播: 矩形窗口近似梯度, 对应 TSSD 论文 Eq. (4)

    ∂S/∂H ≈ (1/a) * sign(|H - ϑ| < a/2)
    """

    @staticmethod
    def forward(ctx, membrane, threshold):
        ctx.save_for_backward(membrane)
        ctx.threshold = threshold
        return (membrane >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        membrane, = ctx.saved_tensors
        threshold = ctx.threshold
        # a = 1.0, 对应论文中的矩形窗口宽度
        grad = (torch.abs(membrane - threshold) < 0.5).float()
        return grad * grad_output, None


class LIFNeuron(nn.Module):
    """
    离散 Leaky Integrate-and-Fire (LIF) 神经元

    对应 TSSD 论文 Eq. (1)-(3):
      H(t) = (1 - 1/τ) * H(t-1) + I(t)     # 膜电位积分
      S(t) = Θ(H(t) - ϑ)                     # 脉冲发放
      H(t) = H(t) - S(t) * ϑ                 # 软重置

    Args:
        tau: 膜电位时间常数 τ, 控制泄漏速率
        threshold: 发放阈值 ϑ
    """

    def __init__(self, tau: float = 2.0, threshold: float = 1.0):
        super().__init__()
        self.tau = tau
        self.threshold = threshold

    def forward(self, x: Tensor, mem: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: 输入电流 I(t), shape (N, dim)
            mem: 上一时刻膜电位 H(t-1), 若为 None 则初始化为零
        Returns:
            spike: 脉冲输出 S(t), shape (N, dim), 取值 {0, 1}
            mem: 重置后的膜电位 H(t), shape (N, dim)
        """
        if mem is None:
            mem = torch.zeros_like(x)
        # Eq. (1): 膜电位积分 (泄漏 + 输入)
        mem = (1.0 - 1.0 / self.tau) * mem + x
        # Eq. (2): 使用代理梯度的脉冲发放
        spike = SurrogateSpike.apply(mem, self.threshold)
        # Eq. (3): 软重置
        mem = mem - spike * self.threshold
        return spike, mem


class SpikingEncoder(nn.Module):
    """
    脉冲时序编码器 — 将 GCN-II 的静态图特征转化为脉冲时序表示

    核心思路: GCN-II 输出作为"输入电流"送入 LIF 神经元, 运行 T 个时间步,
    不同输入幅值的神经元会以不同速率发放脉冲, 产生率编码 (rate coding) 表示.
    最终输出为 T 个时间步脉冲的平均值.

    该编码器使时间自蒸馏 (TSD) 成为可能:
    - T_t 步 (更多步) 的输出更稳定 → 作为"教师"
    - T_s 步 (更少步) 的输出方差更大 → 作为"学生"

    Args:
        dim: 输入/输出特征维度
        tau: LIF 时间常数
        threshold: LIF 发放阈值
        dropout: Dropout 比率
    """

    def __init__(self, dim: int, tau: float = 2.0, threshold: float = 1.0,
                 dropout: float = 0.1):
        super().__init__()
        self.spike_proj = nn.Linear(dim, dim)
        self.spike_bn = nn.BatchNorm1d(dim)
        self.lif = LIFNeuron(tau=tau, threshold=threshold)
        self.output_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, T: int = 4) -> Tensor:
        """
        Args:
            x: GCN-II 输出, shape (N, dim)
            T: 模拟时间步数
        Returns:
            脉冲编码后的特征, shape (N, dim), 包含残差连接
        """
        x_input = self.spike_bn(self.spike_proj(x))

        # 运行 T 个时间步, 收集脉冲输出
        spike_outputs = []
        mem = torch.zeros_like(x_input)
        for t in range(T):
            spike, mem = self.lif(x_input, mem)
            spike_outputs.append(spike)

        # 时间维度平均 → 率编码表示
        avg_spike = torch.stack(spike_outputs, dim=0).mean(dim=0)

        # 投影 + 残差连接 (保留原始图特征信息)
        out = self.dropout(self.output_proj(avg_spike))
        return out + x


class WeakDecoder(nn.Module):
    """
    空间自蒸馏 (SSD) 的弱解码器

    对应 TSSD 论文 Section 3.3:
    在骨干网络中间插入一个轻量级弱分类器, 它基于骨干特征直接做预测.
    最终模型的完整预测作为"教师"指导弱解码器的"学生"预测.

    关键设计:
    - 仅在训练时使用, 推理时丢弃, 不影响推理效率
    - 蒸馏梯度只回传到骨干网络 (GCN-II + Transformer), 不影响 Policy Head
    - 缩短了梯度传播路径, 缓解深层 GCN-II (20层) 的梯度消失

    Args:
        input_dim: 骨干输出的融合特征维度 (fused_dim)
        hidden_dim: 弱解码器的隐藏层维度
        output_dim: 节点嵌入维度 (与 InnerProductDecoder 兼容)
        num_dis: 疾病数量
        dropout: Dropout 比率
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_dis: int, dropout: float = 0.1):
        super().__init__()
        self.num_dis = num_dis
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        # 轻量级内积解码器
        self.weight = nn.Parameter(torch.empty(output_dim, output_dim))
        nn.init.xavier_uniform_(self.weight, gain=1.414)

    def forward(self, residual: Tensor) -> Tensor:
        """
        Args:
            residual: 骨干网络输出, shape (num_dis + num_meta, fused_dim)
        Returns:
            弱预测矩阵, shape (num_meta, num_dis)
        """
        embeddings = self.fc(residual)
        embeddings = F.dropout(embeddings, p=0.1, training=self.training)
        dis_emb = embeddings[:self.num_dis, :]
        meta_emb = embeddings[self.num_dis:, :]
        meta_emb = torch.mm(meta_emb, self.weight)
        output = torch.mm(meta_emb, dis_emb.t())
        return torch.sigmoid(output)

