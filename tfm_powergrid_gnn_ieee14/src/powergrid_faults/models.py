from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import GATv2Conv


def repeat_edge_index(edge_index: torch.Tensor, copies: int, n_nodes: int) -> torch.Tensor:
    """Repeat one graph for `copies` disconnected graph instances."""
    e = edge_index.shape[1]
    offsets = torch.arange(copies, device=edge_index.device).repeat_interleave(e) * n_nodes
    return edge_index.repeat(1, copies) + offsets.unsqueeze(0)


class Heads(nn.Module):
    def __init__(self, hidden: int, dropout: float, include_line: bool = True):
        super().__init__()
        self.shared = nn.Sequential(nn.LayerNorm(hidden), nn.Dropout(dropout))
        self.fault_type = nn.Linear(hidden, 4)
        self.position = nn.Linear(hidden, 3)
        self.security = nn.Linear(hidden, 5)
        self.line = nn.Linear(hidden, 16) if include_line else None

    def forward(self, h):
        h = self.shared(h)
        out = {
            "fault_type": self.fault_type(h),
            "position": self.position(h),
            "security": self.security(h),
        }
        if self.line is not None:
            out["line"] = self.line(h)
        return out


class EdgeLineHead(nn.Module):
    """Score each physical faultable transmission line from its endpoint embeddings.

    A small learned line-ID embedding is included because the PowerFactory IEEE-14
    representation contains two parallel 1-2 circuits with identical endpoints.
    """
    def __init__(self, hidden: int, line_pairs: list[tuple[int, int]], dropout: float):
        super().__init__()
        pairs = torch.tensor(line_pairs, dtype=torch.long)
        self.register_buffer("line_pairs", pairs)
        emb_dim = max(8, hidden // 4)
        self.line_id = nn.Embedding(len(line_pairs), emb_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden * 4 + emb_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, node_h: torch.Tensor) -> torch.Tensor:
        # node_h [B,N,H], line_pairs [L,2]
        u = self.line_pairs[:, 0]
        v = self.line_pairs[:, 1]
        hu = node_h[:, u, :]
        hv = node_h[:, v, :]
        pair = torch.cat([hu, hv, torch.abs(hu - hv), hu * hv], dim=-1)
        ids = torch.arange(self.line_pairs.size(0), device=node_h.device)
        line_e = self.line_id(ids).unsqueeze(0).expand(node_h.size(0), -1, -1)
        return self.scorer(torch.cat([pair, line_e], dim=-1)).squeeze(-1)


class STGAT(nn.Module):
    """Spatial GATv2 encoder per feature block + temporal GRU encoder.

    Input shape: [batch, 14 dynamic blocks, 14 buses, node_features].
    Only buses 1,2,3,6,8 are directly observed; an observation-mask feature makes
    that partial observability explicit. The graph contains transmission lines and
    transformers from the PowerFactory-style IEEE-14 topology.
    """
    def __init__(self, in_features: int, n_buses: int = 14, hidden: int = 64, heads: int = 4,
                 dropout: float = 0.2, temporal_layers: int = 1,
                 fault_line_pairs: list[list[int]] | list[tuple[int, int]] | None = None,
                 edge_localization_head: bool = False, **_):
        super().__init__()
        self.n_buses = n_buses
        self.edge_localization_head = bool(edge_localization_head)
        line_pairs = [tuple(map(int, p)) for p in (fault_line_pairs or [])]
        if self.edge_localization_head and len(line_pairs) != 16:
            raise ValueError("edge_localization_head requires the 16 verified fault_line_pairs")

        # Disable automatic self-loops so returned attention weights correspond exactly
        # to the physical topology. Residual projections retain each node's own signal.
        self.gat1 = GATv2Conv(in_features, hidden, heads=heads, concat=True,
                              dropout=dropout, add_self_loops=False)
        self.res1 = nn.Linear(in_features, hidden * heads)
        self.gat2 = GATv2Conv(hidden * heads, hidden, heads=1, concat=False,
                              dropout=dropout, add_self_loops=False)
        self.res2 = nn.Linear(hidden * heads, hidden)
        self.norm1 = nn.LayerNorm(hidden * heads)
        self.norm2 = nn.LayerNorm(hidden)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden, hidden, num_layers=temporal_layers, batch_first=True,
                          dropout=dropout if temporal_layers > 1 else 0.0)
        # By default line IDs are classified from the graph embedding because the released
        # Excel does not itself document lname -> physical endpoints. The endpoint-aware
        # edge head can be enabled once that mapping is independently verified.
        self.heads = Heads(hidden, dropout, include_line=not self.edge_localization_head)
        self.line_head = EdgeLineHead(hidden, line_pairs, dropout) if self.edge_localization_head else None

    def forward(self, x, edge_index, return_attention=False):
        b, t, n, f = x.shape
        if n != self.n_buses:
            raise ValueError(f"Expected {self.n_buses} buses, got {n}")
        flat = x.reshape(b * t * n, f)
        bei = repeat_edge_index(edge_index, b * t, n)

        h1 = self.gat1(flat, bei) + self.res1(flat)
        h1 = self.drop(self.act(self.norm1(h1)))
        if return_attention:
            h2_msg, (aei, alpha) = self.gat2(h1, bei, return_attention_weights=True)
        else:
            h2_msg = self.gat2(h1, bei)
            aei, alpha = None, None
        h2 = self.act(self.norm2(h2_msg + self.res2(h1)))

        h2 = h2.reshape(b, t, n, -1).permute(0, 2, 1, 3).contiguous()
        temporal, _ = self.gru(h2.reshape(b * n, t, -1))
        node_h = temporal[:, -1].reshape(b, n, -1)
        graph_h = node_h.mean(dim=1)

        out = self.heads(graph_h)
        if self.edge_localization_head:
            out["line"] = self.line_head(node_h)
        out["node_embedding"] = node_h

        if return_attention:
            # Physical directed edge count is unchanged because add_self_loops=False.
            e0 = edge_index.shape[1]
            copy_idx = t - 1  # first sample, final dynamic block
            start = copy_idx * e0
            out["attention_edge_index"] = aei[:, start:start + e0] % n
            a = alpha[start:start + e0]
            out["attention"] = a.mean(dim=-1) if a.ndim > 1 else a
        return out


class CNN1D(nn.Module):
    def __init__(self, in_features: int, n_buses: int = 14, hidden: int = 64,
                 dropout: float = 0.2, **_):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_buses * in_features, hidden, 5, padding=2),
            nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.BatchNorm1d(hidden), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout),
        )
        self.heads = Heads(hidden, dropout, include_line=True)

    def forward(self, x, edge_index=None, return_attention=False):
        z = x.permute(0, 2, 3, 1).reshape(x.size(0), -1, x.size(1))
        return self.heads(self.net(z))


class LSTMBaseline(nn.Module):
    def __init__(self, in_features: int, n_buses: int = 14, hidden: int = 64,
                 dropout: float = 0.2, temporal_layers: int = 1, **_):
        super().__init__()
        self.lstm = nn.LSTM(
            n_buses * in_features, hidden, num_layers=temporal_layers, batch_first=True,
            dropout=dropout if temporal_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.heads = Heads(hidden, dropout, include_line=True)

    def forward(self, x, edge_index=None, return_attention=False):
        z = x.reshape(x.size(0), x.size(1), -1)
        out, _ = self.lstm(z)
        return self.heads(self.drop(out[:, -1]))


def build_model(name: str, **kwargs):
    name = name.lower()
    if name == "stgat":
        return STGAT(**kwargs)
    if name == "cnn":
        return CNN1D(**kwargs)
    if name == "lstm":
        return LSTMBaseline(**kwargs)
    raise ValueError(f"Unknown model: {name}")
