import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ModelAyarlari:
    vocab_size: int = 65536
    dim: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4
    max_seq_len: int = 2048
    num_experts: int = 8
    top_k: int = 2
    expert_dim: int = 1024
    norm_eps: float = 1e-5

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def rope_hesapla(dim: int, max_seq_len: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)

def rope_uygula(xq, xk, cos, sin):
    seq_len = xq.shape[1]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(2)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(2)
    xq_r, xq_i = xq.float().reshape(*xq.shape[:-1], -1, 2).unbind(-1)
    xk_r, xk_i = xk.float().reshape(*xk.shape[:-1], -1, 2).unbind(-1)
    xq_out = torch.stack([xq_r * cos - xq_i * sin, xq_r * sin + xq_i * cos], dim=-1).flatten(3)
    xk_out = torch.stack([xk_r * cos - xk_i * sin, xk_r * sin + xk_i * cos], dim=-1).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class GQADikkat(nn.Module):
    def __init__(self, ayar: ModelAyarlari):
        super().__init__()
        self.n_heads = ayar.n_heads
        self.n_kv_heads = ayar.n_kv_heads
        self.head_dim = ayar.dim // ayar.n_heads
        self.rep = self.n_heads // self.n_kv_heads
        self.wq = nn.Linear(ayar.dim, ayar.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(ayar.dim, ayar.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(ayar.dim, ayar.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(ayar.n_heads * self.head_dim, ayar.dim, bias=False)
    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
        q, k = rope_uygula(q, k, cos, sin)
        k = k.repeat_interleave(self.rep, dim=2).transpose(1, 2)
        v = v.repeat_interleave(self.rep, dim=2).transpose(1, 2)
        q = q.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, C))

class SwiGLUUzman(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoEKatmani(nn.Module):
    def __init__(self, ayar: ModelAyarlari):
        super().__init__()
        self.num_experts = ayar.num_experts
        self.top_k = ayar.top_k
        self.router = nn.Linear(ayar.dim, ayar.num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLUUzman(ayar.dim, ayar.expert_dim) for _ in range(ayar.num_experts)])
    def forward(self, x):
        B, T, C = x.shape
        x_flat = x.view(-1, C)
        router_logits = self.router(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1)
        top_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        expert_usage = routing_weights.mean(dim=0)
        aux_loss = (expert_usage * expert_usage).sum() * self.num_experts
        final_output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (selected_experts == i)
            if mask.any():
                token_idx, k_idx = torch.where(mask)
                final_output[token_idx] += expert(x_flat[token_idx]) * top_weights[token_idx, k_idx].unsqueeze(-1)
        return final_output.view(B, T, C), aux_loss

class TransformerBlok(nn.Module):
    def __init__(self, ayar: ModelAyarlari):
        super().__init__()
        self.attn_norm = RMSNorm(ayar.dim, ayar.norm_eps)
        self.attn = GQADikkat(ayar)
        self.ffn_norm = RMSNorm(ayar.dim, ayar.norm_eps)
        self.moe = MoEKatmani(ayar)
    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        moe_out, aux_loss = self.moe(self.ffn_norm(x))
        return x + moe_out, aux_loss

class TurkceMoEModel(nn.Module):
    def __init__(self, ayar: ModelAyarlari):
        super().__init__()
        self.ayar = ayar
        self.tok_embeddings = nn.Embedding(ayar.vocab_size, ayar.dim)
        self.layers = nn.ModuleList([TransformerBlok(ayar) for _ in range(ayar.n_layers)])
        self.norm = RMSNorm(ayar.dim, ayar.norm_eps)
        self.output = nn.Linear(ayar.dim, ayar.vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight
        cos, sin = rope_hesapla(ayar.dim // ayar.n_heads, ayar.max_seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
    def forward(self, idx, targets=None):
        x = self.tok_embeddings(idx)
        total_aux_loss = 0.0
        for layer in self.layers:
            x, aux_loss = layer(x, self.cos, self.sin)
            total_aux_loss += aux_loss
        logits = self.output(self.norm(x))
        loss = None
        if targets is not None:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = ce_loss + 0.01 * (total_aux_loss / len(self.layers))
        return logits, loss
