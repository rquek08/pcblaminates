"""Shared local fonts for the app, charts and isolated animation documents."""
import base64
from pathlib import Path
import re

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

BODY_FONT = '"DM Sans", sans-serif'
HEADING_FONT = '"DM Serif Display", Georgia, serif'
NUMBER_FONT = '"JetBrains Mono", monospace'
FONT_LINK = '<link rel="stylesheet" href="../static/typography.css">'


@st.cache_data(show_spinner=False)
def font_css():
    """Embed local fonts so srcdoc frames also work without static serving."""
    directory = Path(__file__).parent / "static"
    css = (directory / "typography.css").read_text(encoding="utf-8")

    def embed(match):
        data = base64.b64encode((directory / match[1]).read_bytes()).decode("ascii")
        return f'url("data:font/ttf;base64,{data}")'

    return re.sub(r'url\("(fonts/[^"\n]+)"\)', embed, css)


def document_fonts(html):
    return html.replace(FONT_LINK, "<style>" + font_css() + "</style>")


def configure_charts():
    """Body labels and monospace values, including canvas-rendered Plotly text."""
    axis = dict(tickfont=dict(family=NUMBER_FONT), title=dict(font=dict(family=BODY_FONT)))
    pio.templates["pcb_typography"] = go.layout.Template(
        layout=dict(font=dict(family=BODY_FONT), title=dict(font=dict(family=HEADING_FONT)),
                    xaxis=axis, yaxis=axis,
                    hoverlabel=dict(font=dict(family=NUMBER_FONT)),
                    scene=dict(xaxis=axis, yaxis=axis, zaxis=axis)),
        data=dict(bar=[go.Bar(textfont=dict(family=NUMBER_FONT))]),
    )
    pio.templates.default = "plotly+pcb_typography"
