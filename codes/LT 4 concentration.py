"""
This script visualises the relationship between the chi_626 mismatch and the Dp17O values.

INPUT:
- LT_Table_S3.csv: A CSV file containing data from the University of Cape Town.
- LT_Table_S4.csv: A CSV file containing data from the University of Göttingen.

OUTPUT:
- LT_Figure_3.png
- LT_Figure_3b.png (optional, currently commented out)
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
c_NBS18, c_IAEA603 = "#F75056", "#814997"
c_light, c_heavy = "#309FD1", "#0C3992"
c_zero = "#EE7BAE"

# Get data - University of Cape Town
df_uct = pd.read_csv(os.path.join(data_dir, "LT_Table_S3.csv"))
df_uct['pCO2Mismatch'] = df_uct['Xp626_L2_mean']/1000 - 421
df_uct['pCO2Mismatch_error'] = df_uct['Xp626_L2_std']/1000
df_uct["Dp17O"] = df_uct['Dp17O_raw_mean']*1000
df_uct["Dp17OError"] = (df_uct['Dp17O_raw_std']*1000)/ np.sqrt(df_uct['NumCycles'])
df_uct['d18O'] = unprime(df_uct['dp18O_L2_raw_mean'])
df_uct['d18OError'] = df_uct['dp18O_L2_raw_std'] / np.sqrt(df_uct['NumCycles'])
df_uct_zero_2025 = df_uct[df_uct['SampleName'].str.contains("ZERO") & (df_uct['Session'] == 2025)].copy()
df_uct_603_2024 = df_uct[df_uct['SampleName'].str.contains("603") & (df_uct['Session'] == 2024)].copy()
df_uct_603_2025 = df_uct[df_uct['SampleName'].str.contains("603") & (df_uct['Session'] == 2025)].copy()

# Get data - University of Göttingen
df_ug = pd.read_csv(os.path.join(data_dir, "LT_Table_S4.csv"))
df_ug['SampleName'] = df_ug['SampleName'].str.replace("VsRef", " $CO_2$")
df_ug['dateTimeMeasured'] = pd.to_datetime(df_ug['dateTimeMeasured'], format='%Y-%m-%d %H:%M:%S')

# Print the start and end of analytical sessions in Göttingen
# print("Göttingen analytical sessions:")
# years = df_ug['dateTimeMeasured'].dt.year.unique()
# for year in years:
#     earliest = df_ug[df_ug['dateTimeMeasured'].dt.year == year]['dateTimeMeasured'].min()
#     latest = df_ug[df_ug['dateTimeMeasured'].dt.year == year]['dateTimeMeasured'].max()
#     print(f"{earliest.date().strftime('%d %B, %Y')} to {latest.date().strftime('%d %B, %Y')}")

# Convert delta-values relative to WG (so a zero enrichment would yield a d18O of 0)
df_ug["d18O"] = deltaO_vs_WG(df_ug["d18O"], 28.048)
df_ug["d17O"] = deltaO_vs_WG(df_ug["d17O"], 14.621)
df_ug["Dp17O"] = Dp17O(df_ug["d17O"], df_ug["d18O"])

df_ug_light_2023 = df_ug[
    df_ug['SampleName'].str.contains("light") &
    (df_ug['dateTimeMeasured'].dt.year == 2023)
].copy()

df_ug_light_2025 = df_ug[
    df_ug['SampleName'].str.contains("light") &
    (df_ug['dateTimeMeasured'].dt.year == 2025)
].copy()

df_ug_heavy_2023 = df_ug[
    df_ug['SampleName'].str.contains("heavy") &
    (df_ug['dateTimeMeasured'].dt.year == 2023)
].copy()

df_ug_heavy_2025 = df_ug[
    df_ug['SampleName'].str.contains("heavy") &
    (df_ug['dateTimeMeasured'].dt.year == 2025)
].copy()


#  Plot ∆17O vs. X626 mismatch

dataset = [
    (df_ug_light_2023['pCO2Mismatch'], df_ug_light_2023['Dp17O'],
     df_ug_light_2023['pCO2Mismatch_error'], df_ug_light_2023['Dp17OError'],
     c_light, "-", "light CO$_{2}$"),

    (df_ug_heavy_2023['pCO2Mismatch'], df_ug_heavy_2023['Dp17O'],
     df_ug_heavy_2023['pCO2Mismatch_error'], df_ug_heavy_2023['Dp17OError'],
     c_heavy, "-", "heavy CO$_{2}$"),

    (df_ug_light_2025['pCO2Mismatch'], df_ug_light_2025['Dp17O'],
     df_ug_light_2025['pCO2Mismatch_error'], df_ug_light_2025['Dp17OError'],
     c_light, "--", "light CO$_{2}$"),

    (df_ug_heavy_2025['pCO2Mismatch'], df_ug_heavy_2025['Dp17O'],
     df_ug_heavy_2025['pCO2Mismatch_error'], df_ug_heavy_2025['Dp17OError'],
     c_heavy, "--", "heavy CO$_{2}$"),

    (df_uct_603_2024['pCO2Mismatch'], df_uct_603_2024['Dp17O'],
     df_uct_603_2024['pCO2Mismatch_error'], df_uct_603_2024['Dp17OError'],
     c_IAEA603, ":", "IAEA-603"),

    (df_uct_603_2025['pCO2Mismatch'], df_uct_603_2025['Dp17O'],
     df_uct_603_2025['pCO2Mismatch_error'], df_uct_603_2025['Dp17OError'],
     c_IAEA603, "-", "IAEA-603"),

    (df_uct_zero_2025['pCO2Mismatch'], df_uct_zero_2025['Dp17O'],
     df_uct_zero_2025['pCO2Mismatch_error'], df_uct_zero_2025['Dp17OError'],
     c_zero, "-", "zero-enrichment"),
]

fig, ax = plt.subplots()

def plot_dataset(ax, dataset):
    for x, y, xerr, yerr, color, ls, label in dataset:
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        
        ax.plot(x_line, slope * x_line + intercept,
                color=color, ls=ls, lw=2, zorder=-1,
                label=f"{label}\n$\it{{m}}$ = {slope:.1f}$\pm${std_err:.1f}, $\it{{N}}$={len(x)}")


        # Uncomment the following lines to plot the data points with error bars:
        # ax.scatter(x, y,
        #            ec=color, fc=f"{color}80", marker="o")
        # ax.errorbar(
        #     x, y,
        #     xerr=xerr, yerr=yerr,
        #     fmt="none",
        #     ecolor=f"{color}80",
        #     elinewidth=0.8,
        # )

        ax.legend(
            loc="upper left",
            bbox_to_anchor=(1, 1.02),
            fontsize=7,
            markerscale=0.9
        )


plot_dataset(ax, dataset)

handles, labels = ax.get_legend_handles_labels()

# Create dummy heading handles
from matplotlib.lines import Line2D
heading1 = Line2D([], [], color='none', label=r'$\bf{Göttingen\ 2023}$')
heading2 = Line2D([], [], color='none', label=r'$\bf{Göttingen\ 2025}$')
heading3 = Line2D([], [], color='none', label=r'$\bf{Cape\ Town\ 2023}$')
heading4 = Line2D([], [], color='none', label=r'$\bf{Cape\ Town\ 2024}$')

# Insert headings at desired positions
new_handles = []
new_labels = []

for idx, (h, l) in enumerate(zip(handles, labels)):
    if idx == 0:
        new_handles.append(heading1)
        new_labels.append(heading1.get_label())
    if idx == 2:
        new_handles.append(heading2)
        new_labels.append(heading2.get_label())
    if idx == 4:
        new_handles.append(heading3)
        new_labels.append(heading3.get_label())
    if idx == 5:
        new_handles.append(heading4)
        new_labels.append(heading4.get_label())
    
    new_handles.append(h)
    new_labels.append(l)

# Redraw the legend
ax.legend(new_handles, new_labels,
          loc="upper left",
          bbox_to_anchor=(1, 1.02),
          fontsize=7,
          markerscale=0.9)

ax.set_xlabel(r"$\chi\prime^{smp}_{626, meas} - \chi\prime^{wg}_{626, meas} $ ($\mu$mol mol$^{-1}$)")
ax.set_ylabel(r"$\Delta\prime^{17}O^{smp/wg}_{meas}$ (ppm)")

plt.savefig(os.path.join(figures_dir, "LT_Figure_3.png"))
plt.close("all")


#  The following lines calculate the X626 mismatch dependence of d18O
dataset = [
    (df_ug_light_2023['pCO2Mismatch'], df_ug_light_2023['d18O'],
     "light CO$_{2}$"),

    (df_ug_heavy_2023['pCO2Mismatch'], df_ug_heavy_2023['d18O'],
     "heavy CO$_{2}$"),

    (df_ug_light_2025['pCO2Mismatch'], df_ug_light_2025['d18O'],
     "light CO$_{2}$"),

    (df_ug_heavy_2025['pCO2Mismatch'], df_ug_heavy_2025['d18O'],
     "heavy CO$_{2}$"),

    (df_uct_603_2024['pCO2Mismatch'], df_uct_603_2024['d18O'],
     "IAEA-603"),

    (df_uct_603_2025['pCO2Mismatch'], df_uct_603_2025['d18O'],
     "IAEA-603"),

    (df_uct_zero_2025['pCO2Mismatch'], df_uct_zero_2025['d18O'],
     "zero-enrichment"),
]

for x, y, label in dataset:
    xy = pd.concat([x, y], axis=1).dropna()
    if xy.empty or len(xy) < 2:
        print(f"{label}: not enough valid points for regression")
    
    xvals = xy.iloc[:, 0].to_numpy()
    yvals = xy.iloc[:, 1].to_numpy()
    mask = np.isfinite(xvals) & np.isfinite(yvals)
    if mask.sum() < 2:
        print(f"{label}: not enough finite points for regression")

    slope, intercept, r_value, p_value, std_err = linregress(xvals[mask], yvals[mask])
    print(f"{label}: slope {slope:.3f} ± {std_err:.3f} per umol mol-1")