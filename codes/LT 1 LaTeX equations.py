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
    r"I_{(\nu)} = I_{o\,(\nu)} \exp\left[-\frac{N_{A}}{\ln(10)}\;S_{(T)}\;\phi_{(\nu,P,T)}\;L\;C \right]",
    
    # Equation 3
    r"\chi^{\prime}_{\mathrm{i}} = C_{\mathrm{i}} \; R \; \frac{T}{P} \; \frac{1}{X_{{\mathrm{i}}}}",
    
    # Equation 4
    r"\delta^{17}\mathrm{O}_{\mathrm{meas}} = \frac{{\chi}^{\prime}_{627\mathrm{,meas}}}{{{\chi}^{\prime}_{626\mathrm{,meas}}}} - 1",
    
    # Equation 5
    r"\delta^{17}\mathrm{O}^{\mathrm{smp/wg}}_{\mathrm{meas}} = \frac{\delta^{17}\mathrm{O}^{\mathrm{smp}}_{\mathrm{meas}}+1}{\delta^{17}\mathrm{O}^{\mathrm{wg}}_{\mathrm{meas}}+1} - 1",
    
    # Equation 6
    r"{\chi}^{\prime}_{627\mathrm{,meas}} = a_{627} \times \chi^{\prime}_{627\mathrm{,true}} + b_{627}",
    
    # Equation 7
    r"\delta^{17}\mathrm{O}_{\mathrm{meas}} = \frac{a_{627} \times \chi^{\prime}_{627\mathrm{,true}} + b_{627}}{{a_{626} \times \chi^{\prime}_{626\mathrm{,true}} + b_{626}}} - 1",
    
    # Equation 8
    r"\delta^{17}\mathrm{O}_{\mathrm{true}}=\frac{\chi^{\prime}_{626\mathrm{,meas}}}{\frac{a_{627}}{a_{626}}\;(\chi^{\prime}_{626\mathrm{,meas}}-b_{626})}\Big(\delta^{17}\mathrm{O_{\mathrm{meas}}}+\frac{\frac{a_{627}}{a_{626}} \times b_{626}-b_{627}}{\chi^{\prime}_{626\mathrm{,meas}}}-\frac{a_{627}}{a_{626}}+1\Big)",

    # Equation 9
    r"\Delta^{\prime17}\mathrm{O}^{\mathrm{smp/wg}}_{\mathrm{true}} \simeq  \Delta^{\prime17}\mathrm{O}^{\mathrm{smp/wg}}_{\mathrm{meas}} - m  \times \left( \chi^{\prime\mathrm{smp}}_{626\mathrm{,meas}} - \chi^{\prime\mathrm{wg}}_{626\mathrm{,meas}} \right) ",

    # Equation A1
    r"\delta^{17}\mathrm{O}_{\mathrm{true}}=\frac{\chi^{\prime}_{626\mathrm{,meas}}}{A_{627}\;(\chi^{\prime}_{626\mathrm{,meas}}-b_{626})}\Big(\delta^{17}\mathrm{O_{\mathrm{meas}}}+\frac{A_{627} \times b_{626}-b_{627}}{\chi^{\prime}_{626\mathrm{,meas}}}-A_{627}+1\Big)",

    # Equation A2
    r"\delta^{17}\mathrm{O}_{\mathrm{true}} \simeq \frac{\chi^{\prime}_{626\mathrm{,meas}}} {A_{627} (\chi^{\prime}_{626\mathrm{,meas}}-b_{626})} \delta^{17}\mathrm{O}_{\mathrm{meas}}",

    # Equation A3
    r"\Delta^{\prime17}\mathrm{O}^{\mathrm{smp/wg}}_{\mathrm{true}} = \Delta^{\prime17}\mathrm{O}^{\mathrm{smp}}_{\mathrm{true}} - \Delta^{\prime17}\mathrm{O}^{\mathrm{wg}}_{\mathrm{true}}",

    # Equation A4
    r"\Delta^{\prime17}\mathrm{O^{smp/wg}_{true}} = \Delta^{\prime17}\mathrm{O^{smp/wg}_{meas}}+(1-\lambda_{\mathrm{RL}})\ln\Big[\frac{\chi^{\prime\mathrm{smp}}_{626\mathrm{,meas}}(\chi^{\prime\mathrm{wg}}_{626\mathrm{,meas}}-b_{626})}{\chi^{\prime\mathrm{wg}}_{626\mathrm{,meas}}(\chi^{\prime\mathrm{smp}}_{626\mathrm{,meas}}-b_{626})}\Big]",

    # Equation A5
    r"\Delta^{\prime17}\mathrm{O^{true}_{smp/wg}} \simeq \Delta^{\prime17}\mathrm{O^{meas}_{smp/wg}}+(1-\lambda_{\mathrm{RL}})b_{626}\Big[\frac{1}{\chi^{\prime\mathrm{smp}}_{626\mathrm{,meas}}}-\frac{1}{\chi^{\prime\mathrm{wg}}_{626\mathrm{,meas}}}\Big]",

    # Equation A6
    r"\Delta^{\prime17}\mathrm{O^{true}_{smp/wg}} \simeq \Delta^{\prime17}\mathrm{O^{meas}_{smp/wg}}-(1-\lambda_{\mathrm{RL}})b_{626}/(\chi^{\prime\mathrm{wg}}_{626\mathrm{,meas}})^2\times(\chi^{\prime\mathrm{smp}}_{626\mathrm{,meas}}-\chi^{\prime\mathrm{wg}}_{626\mathrm{,meas}})",

]

for i, eq in enumerate(equations, 1):
    latex = r"\dpi{300}\bg{white} " + eq
    encoded = quote(latex)
    url = f"https://latex.codecogs.com/png.image?{encoded}"
    response = requests.get(url)

    eqn = i if i <= 9 else f"A{i-9}"

    if response.status_code == 200:
        output_file = os.path.join(figures_dir, f"LT_Equation_{eqn}.png")
        with open(output_file, "wb") as f:
            f.write(response.content)

        # Calculate image height in cm to match a font size of 12 pt in Word
        with Image.open(output_file) as img:
            height_px = img.height
        target_cm = height_px * (0.45 / 55)
        print(f"Equation {eqn}: set height to {target_cm:.2f} cm")
    else:
        print(f"Failed to render Equation {i}: {response.status_code}")
