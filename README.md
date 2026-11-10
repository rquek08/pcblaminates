[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pcblaminates.streamlit.app/)
# Bonded By Stress

An interactive engineering and screening tool built to evaluate, decouple, and visualise hygrothermal mismatch stresses at copper-clad laminate (CCL) interfaces. 

### Overview

In printed circuit board (PCB) architectures, moisture absorption and thermal expansions drive competing deformation modes:

1. Hygroscopic swelling: The polymer dielectric core absorbs ambient water, causing matrix expansion, whereas copper foils are impermeable to moisture.
2. Thermal mismatch strain: Different coefficients of thermal expansion (CTE) between the copper foil and dielectric substrate induce different in-plane expansion.
   
When these effects combine, they generate interfacial shear stresses that increase the risk of delamination. This platform provides an analytical screening pipeline to evaluate the in-plane stress indicator, rank parameter dominance via variance-based sensitivity analysis, and explore candidate material design windows.

### Features

1. Stress Calculation: 
Select a laminate preset from the materials library or customise your own to calculate the moisture and temperature induced strain, as well as a combined stress index. This gives users a sense of whether moisture or thermal contributions dominate in inducing stress on the laminate-copper interface.

2. Sensitivity Analysis: 
Evaluate First-Order (S1) and Total-Order (ST) Sobol indices using Saltelli sampling to quantify how much variance each input contributes to interfacial stress. Users are able to isolate exposure conditions or materials properties depending on which information is more useful for their application.

3. Inverse Exploration: 
Specify an allowable stress, or other material parameters thresholds, to identify permissible candidate materials from the database. This provides users with suggestions on possible laminates that they can use for their application.
