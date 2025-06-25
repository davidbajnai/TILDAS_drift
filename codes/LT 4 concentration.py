"""
This script visualises the relationship between the pCO2 mismatch and the Dp17O values.

INPUT:
- LT_Table_S3.csv: A CSV file containing data from the University of Cape Town.
- LT_Table_S4.csv: A CSV file containing data from the University of Göttingen.

OUTPUT:
- LT_Figure_3.png
"""

# Import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from functions import *

# Get script directory for relative file paths
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get data directory
data_dir = os.path.join(script_dir, '../data')

# Get figures directory
figures_dir = os.path.join(script_dir, '../figures')

# Plot parameters
plt.rcParams.update({"font.size": 8})
plt.rcParams["figure.figsize"] = (3, 3)
plt.rcParams["patch.linewidth"] = 1
plt.rcParams['lines.linewidth'] = 1
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams['savefig.transparent'] = False
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['legend.loc'] = 'upper right'

# Colors for plotting the reference materials
c_NBS18, c_IAEA603 = "#F75056", "#4D0820"
c_light, c_heavy = "#309FD1", "#0C3992"
c_zero = "#814997"

# Get data - University of Cape Town
df_uct = pd.read_csv(os.path.join(data_dir, "LT_Table_S3.csv"))
df_uct['pCO2Mismatch'] = df_uct['Xp626_L2_mean']/1000 - 421
df_uct['pCO2Mismatch_error'] = df_uct['Xp626_L2_std']/1000
df_uct["Dp17O"] = df_uct['Dp17O_raw_mean']*1000
df_uct["Dp17OError"] = (df_uct['Dp17O_raw_std']*1000)/ np.sqrt(df_uct['NumCycles'])
df_uct_zero_2025 = df_uct[df_uct['SampleName'].str.contains("ZERO") & (df_uct['Session'] == 2025)].copy()
df_uct_603_2024 = df_uct[df_uct['SampleName'].str.contains("603") & (df_uct['Session'] == 2024)].copy()
df_uct_603_2025 = df_uct[df_uct['SampleName'].str.contains("603") & (df_uct['Session'] == 2025)].copy()

# Get data - University of Göttingen
df_ug = pd.read_csv(os.path.join(data_dir, "LT_Table_S4.csv"))
df_ug['SampleName'] = df_ug['SampleName'].str.replace("VsRef", " $CO_2$")
df_ug['dateTimeMeasured'] = pd.to_datetime(df_ug['dateTimeMeasured'], format='%Y-%m-%d %H:%M:%S')

# Convert d-values relative to WG (so a zero enrichment would yield a d18O of 0)
df_ug["d18O"] = deltaO_vs_WG(df_ug["d18O"], 28.048)
df_ug["d17O"] = deltaO_vs_WG(df_ug["d17O"], 14.621)
df_ug["Dp17O"] = Dp17O(df_ug["d17O"], df_ug["d18O"])

df_ug_light = df_ug[df_ug['SampleName'].str.contains("light")].copy()
df_ug_heavy = df_ug[df_ug['SampleName'].str.contains("heavy")].copy()

#  make a plot ∆17O vs. pCO2

fig, ax = plt.subplots()

datasets = [
    (df_ug_light['pCO2Mismatch'], df_ug_light['Dp17O'], c_light, "light CO$_{2}$ (UG)",
     df_ug_light['pCO2Mismatch_error'], df_ug_light['Dp17OError']),
    
    (df_ug_heavy['pCO2Mismatch'], df_ug_heavy['Dp17O'], c_heavy, "heavy CO$_{2}$ (UG)",
     df_ug_heavy['pCO2Mismatch_error'], df_ug_heavy['Dp17OError']),
    
    (df_uct_zero_2025['pCO2Mismatch'], df_uct_zero_2025['Dp17O'], c_zero, "zero-enrichment (UCT, 2025)",
     df_uct_zero_2025['pCO2Mismatch_error'], df_uct_zero_2025['Dp17OError']),
    
    (df_uct_603_2025['pCO2Mismatch'], df_uct_603_2025['Dp17O'], c_IAEA603, "IAEA-603 (UCT, 2025)",
     df_uct_603_2025['pCO2Mismatch_error'], df_uct_603_2025['Dp17OError']),
    
    (df_uct_603_2024['pCO2Mismatch'], df_uct_603_2024['Dp17O'], c_IAEA603, "IAEA-603 (UCT, 2024)",
     df_uct_603_2024['pCO2Mismatch_error'], df_uct_603_2024['Dp17OError']),
]

for x, y, color, label, xerr, yerr in datasets:
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, color=color, ls="-", lw = 1.2)

    marker = 'v' if '2024' in label and 'IAEA' in label else 'o'

    ax.scatter(x, y,
               ec=color, fc=f"{color}80", marker=marker,
               label=f"{label}\n$\it{{m}}$ = {slope:.1f}$\pm${std_err:.1f}  $\it{{N}}$ = {len(x)}")

    # ax.errorbar(
    #     x, y,
    #     xerr=xerr, yerr=yerr,
    #     fmt="none",
    #     ecolor=f"{color}80",
    #     elinewidth=0.8,
    # )


ax.legend(loc="upper left", bbox_to_anchor=(1, 1.02))

ax.set_xlabel(r"$\chi\prime^{sam}_{626} - \chi\prime^{std}_{626} $ ($\mu$mol mol$^{-1}$)")
ax.set_ylabel(r"$\Delta\prime^{17}O_{smp/std}$ (ppm)")

plt.savefig(os.path.join(figures_dir, "LT_Figure_3.png"))
plt.close("all")