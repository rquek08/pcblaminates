"""Interactive PCB overview and a close-up of the selected core interface."""
import json
from pathlib import Path

import streamlit as st
from typography import document_fonts


def stack_html(material, thickness_mm, laminate_cte, copper_cte, interface):
    config = dict(material=material, thickness=thickness_mm, laminateCTE=laminate_cte,
                  copperCTE=copper_cte, interface=interface)
    # Keep text values inside JSON, then assign them with textContent in the
    # frame. A material name must never become HTML or terminate the script.
    payload = json.dumps(config).replace("<", "\\u003c")
    template = (Path(__file__).parent / "assets" / "pcb_stack_selector.html").read_text(encoding="utf-8")
    return document_fonts(template).replace("/* STACK_CONFIG */", "const config = " + payload + ";")


def render_stack(material, thickness_mm, laminate_cte, copper_cte):
    interface = st.radio("Copper interface", ["Upper core face · L2", "Lower core face · L3"],
                         horizontal=True, key="simulation_core_interface", persist_state="session")
    st.iframe(stack_html(material, thickness_mm, laminate_cte, copper_cte,
                         "upper" if interface.startswith("Upper") else "lower"), height="content")
    st.caption("Illustrative four-layer PCB, not a manufacturer stack-up. The selected material supplies the core properties.")
    st.caption("Current moisture calculation: two exposed laminate faces. Copper barriers and edge entry are not included.")
