"""
Image visualization helpers.

Generates plotly figures for image display and analytics.
"""

from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image


def make_image_figure(img: Image.Image, title: str = "Image Preview") -> go.Figure:
    """Render a PIL Image as a plotly figure."""
    arr = np.array(img)
    if arr.ndim == 2:
        fig = px.imshow(arr, color_continuous_scale="gray", title=title)
    else:
        fig = px.imshow(arr, title=title)
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=40, b=0),
        coloraxis_showscale=False,
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig
