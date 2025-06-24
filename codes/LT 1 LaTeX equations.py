"""
This script renders the LaTeX equations for the manuscript as PNG images using the Codecogs API.
"""

# Import libraries
import requests
import os
from urllib.parse import quote
from PIL import Image

# Get script directory for relative file paths
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get figures directory
figures_dir = os.path.join(script_dir, "../figures") 

# LaTeX equations to render
equations = [

    # Equation 1
    r"\Delta^{\prime17}\mathrm{O} = \ln(\delta^{17}\mathrm{O} + 1) - \lambda_{\mathrm{RL}} \cdot \ln(\delta^{18}\mathrm{O} + 1) - \gamma_{\mathrm{RL}}",
    
    # Equation 2
    r"I_{(\nu)} = I_{o\,(\nu)} \exp\left[-\frac{N_{A}}{ln(10)}\;S_{(T)}\;\phi_{(\nu,P,T)}\;L\;C \right]",
    
    # Equation 3
    r"\chi^{\prime}_{\mathrm{i}} = C_{\mathrm{i}} \; R \; \frac{T}{P} \; \frac{1}{X_{{\mathrm{i}}}}",
    
    # Equation 4
    r"\delta^{17}\mathrm{O}_{\text{meas}} = \frac{{\chi'}_{627}}{{{\chi'}_{626}}} - 1 \mathrm{\;\;and\;\;} \delta^{18}\mathrm{O}_{\text{meas}} =\frac{{\chi'}_{628}}{{{\chi'}_{626}}} - 1",
    
    # Equation 5
    r"\delta^{17}\mathrm{O}_{\mathrm{smp/std}} = \frac{\delta^{17}\mathrm{O}^{\mathrm{meas}}_{\mathrm{smp}}+1}{\delta^{17}\mathrm{O}^{\mathrm{meas}}_{\mathrm{std}}+1} - 1 \mathrm{\;\;and\;\;} \delta^{18}\mathrm{O}_{\mathrm{smp/std}} = \frac{\delta^{18}\mathrm{O}^{\mathrm{meas}}_{\mathrm{smp}}+1}{\delta^{18}\mathrm{O}^{\mathrm{meas}}_{\mathrm{std}}+1} - 1",
    
    # Equation 6
    r"\chi^{\prime \mathrm{true}}_{627} = a_{627} \cdot \chi^{\prime}_{627} + b_{627}",
    
    # Equation 7
    r"\delta^{17}\mathrm{O}_{\text{true}} = \frac{a_{627} \cdot {\chi'}_{627} + b_{627}}{{a_{626} \cdot {\chi'}_{626} + b_{626}}} - 1  \mathrm{\;\;and\;\;} \delta^{18}\mathrm{O}_{\text{true}} =\frac{a_{628} \cdot {\chi'}_{628} + b_{628}}{{a_{626} \cdot {\chi'}_{626} + b_{626}}} - 1",
    
    # Equation 8
    r"\delta^{17}\mathrm{O}_{\text{true}} = \delta^{17}\mathrm{O}_{\text{meas}} \cdot \frac{\chi'_{626} \cdot a_{627}}{\chi'_{626} \cdot a_{626} + b_{626}} + \frac{\chi'_{626} \cdot (a_{627} - a_{626}) + b_{627} - b_{626}}{\chi'_{626} \cdot a_{626} + b_{626}}",

    # Equation 9
    r"\Delta^{\prime17}\mathrm{O}^{\text{true}}_{\text{smp/std}} \simeq  \Delta^{\prime17}\mathrm{O}_{\text{smp/std}} - m  \cdot \left( \chi^{\prime \mathrm{smp}}_{626} - \chi^{\prime \mathrm{std}}_{626} \right) ",
]

for i, eq in enumerate(equations, 1):
    latex = r"\dpi{300}\bg{white} " + eq
    encoded = quote(latex)
    url = f"https://latex.codecogs.com/png.image?{encoded}"
    response = requests.get(url)
    if response.status_code == 200:
        output_file = os.path.join(figures_dir, f"LT_Equation_{i}.png")
        with open(output_file, "wb") as f:
            f.write(response.content)

        # Calculate image height in cm to match a font size of 12 pt in Word
        with Image.open(output_file) as img:
            height_px = img.height
        target_cm = height_px * (0.45 / 55)
        print(f"Equation {i}: set height to {target_cm:.2f} cm")
    else:
        print(f"Failed to render Equation {i}: {response.status_code}")