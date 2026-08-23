from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from .topology import case14_powerfactory_graph, topological_positions

# The function generates a Plotly figure showing the power grid graph, including buses, transformers, and transmission lines, with optional overlays for predicted fault lines, true fault lines, and associated probabilities. 
def _line_midpoint(pos, u, v, fraction=0.5):
    x = pos[u][0] + fraction * (pos[v][0] - pos[u][0])
    y = pos[u][1] + fraction * (pos[v][1] - pos[u][1])
    return x, y

# Generates a Plotly figure visualizing the IEEE-14 power grid topology, including buses, transformers, and transmission lines.
def topology_figure(line_probs=None, true_line=None, pred_line=None,
                    true_pol=None, pred_pol=None, attention=None,
                    title="IEEE-14 — fault-line localization"):
    g, _, meta = case14_powerfactory_graph()
    pos = topological_positions(g)
    probs = np.asarray(line_probs if line_probs is not None else np.zeros(16), dtype=float)
    if probs.size != 16:
        probs = np.resize(probs, 16)

    fig = go.Figure()

    # Transformers are part of message passing but are not fault-line targets in this dataset.
    for u, v in meta["transformers"]:
        fig.add_trace(go.Scatter(
            x=[pos[u][0], pos[v][0]], y=[pos[u][1], pos[v][1]], mode="lines",
            line=dict(width=2, dash="dot"), hovertemplate=f"Transformer {u+1}↔{v+1}<extra></extra>",
            showlegend=False,
        ))

    pmax = max(float(probs.max()), 1e-9)
    for i, ((u, v), name) in enumerate(zip(meta["fault_lines"], meta["line_names"])):
        width = 1.5 + 7.0 * float(probs[i] / pmax)
        fig.add_trace(go.Scatter(
            x=[pos[u][0], pos[v][0]], y=[pos[u][1], pos[v][1]], mode="lines",
            line=dict(width=width),
            hovertemplate=f"Line {i+1}: {name}<br>P={probs[i]:.4f}<extra></extra>",
            showlegend=False,
        ))

    xs = [pos[i][0] for i in range(14)]
    ys = [pos[i][1] for i in range(14)]
    observed = set(meta["generator_buses"])
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text", text=[str(i + 1) for i in range(14)],
        textposition="middle center", hovertext=[
            f"Bus {i+1}<br>{'PMU/generator observed' if i in observed else 'unobserved bus'}"
            for i in range(14)
        ], hoverinfo="text", name="Buses",
        marker=dict(size=[27 if i in observed else 20 for i in range(14)], line=dict(width=2)),
    ))

    if true_line is not None and 0 <= int(true_line) < 16:
        u, v = meta["fault_lines"][int(true_line)]
        frac = float(true_pol) / 100.0 if true_pol is not None else 0.5
        x, y = _line_midpoint(pos, u, v, frac)
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers", name="True fault",
            hovertemplate=f"True: line {int(true_line)+1}<br>POL={float(true_pol):.1f}%<extra></extra>" if true_pol is not None else None,
            marker=dict(size=22, symbol="star"),
        ))

    if pred_line is not None and 0 <= int(pred_line) < 16:
        u, v = meta["fault_lines"][int(pred_line)]
        frac = float(pred_pol) / 100.0 if pred_pol is not None else 0.5
        x, y = _line_midpoint(pos, u, v, frac)
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers", name="Predicted fault",
            hovertemplate=f"Predicted: line {int(pred_line)+1}<br>POL={float(pred_pol):.1f}%<extra></extra>" if pred_pol is not None else None,
            marker=dict(size=20, symbol="x"),
        ))

    fig.update_layout(
        title=title, hovermode="closest", margin=dict(l=10, r=10, t=55, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False), height=620, showlegend=True,
    )
    return fig
