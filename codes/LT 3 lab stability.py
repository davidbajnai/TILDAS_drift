"""
This script compares the stability of the lab setups at the University of Göttingen and the University of Cape Town.

INPUT:
- LT_Table_S1.csv: Contains data from the University of Göttingen.
- LT_Table_S2.csv: Contains data from the University of Cape Town.

OUTPUT:
- LT_Figure_2.png
- LT_Figure_6.png
"""

# Import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functions import *
from tabulate import tabulate
from scipy.stats import linregress

# Get script directory for relative file paths
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get data directory
data_dir = os.path.join(script_dir, '../data')

# Get figures directory
figures_dir = os.path.join(script_dir, '../figures')

# Plot parameters
plt.rcParams.update({"font.size": 8})
plt.rcParams["figure.figsize"] = (6, 7.5)
plt.rcParams["patch.linewidth"] = 1
plt.rcParams['lines.linewidth'] = 1
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams['savefig.transparent'] = False
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['legend.loc'] = 'lower right'

# Get data - University of Cape Town
df_uct = pd.read_csv(os.path.join(data_dir, "LT_Table_S2.csv"))
df_uct['dateTimeMeasured'] = pd.to_datetime(df_uct['Name'], format='%y%m%d_%H%M%S', errors='coerce')
df_uct = df_uct.sort_values(by=["dateTimeMeasured"], ascending=True, ignore_index=True)

# Make column names consistent
df_uct["SampleName"] = df_uct["AnalyticalID"].str.split('-').str[0]
df_uct['Dp17O'] = df_uct['Dp17O_corr']*1000
df_uct['Dp17O_meas'] = df_uct['Dp17O_raw_mean']*1000
df_uct['Dp17O_error'] = (df_uct['Dp17O_raw_std']*1000)/np.sqrt(df_uct['cycle_n'])
df_uct['cell_temperature'] = df_uct['Traw_mean']
df_uct['cell_temperature_error'] = df_uct['Traw_std']
df_uct['cell_pressure'] = torr_to_pascal(df_uct['Praw_mean'])
df_uct['cell_pressure_error'] = torr_to_pascal(df_uct['Praw_std'])
df_uct['chi_p_626'] = df_uct['Xp626_L2_mean']/1000
df_uct['chi_p_626_error'] = df_uct['Xp626_L2_std']/1000
df_uct['d18O'] = unprime(df_uct['dp18O_L2_raw_mean'])
df_uct['d17O'] = unprime(df_uct['dp17O_L2_raw_mean'])

# Exclude some samples based on AnalyticalID
df_uct = df_uct[df_uct['AnalyticalID'] != "IAEA603-27"]
df_uct = df_uct[df_uct['AnalyticalID'] != "IAEA603-33"]
df_uct = df_uct[df_uct['AnalyticalID'] != "IAEA603-35"]
df_uct = df_uct[df_uct['AnalyticalID'] != "IAEA603-36"]
df_uct = df_uct[df_uct['AnalyticalID'] != "IAEA603-59"]
df_uct = df_uct[df_uct['AnalyticalID'] != "NBS18-36"]
df_uct = df_uct[df_uct['AnalyticalID'] != "NBS18-37"]
df_uct = df_uct[df_uct['AnalyticalID'] != "NBS18-56"]
df_uct = df_uct[df_uct['AnalyticalID'] != "NBS18-57"]

# Get data - University of Göttingen
df_ug = pd.read_csv(os.path.join(data_dir, "LT_Table_S1.csv"))
df_ug['SampleName'] = df_ug['SampleName'].str.replace("VsRef", " $CO_2$")
df_ug['dateTimeMeasured'] = pd.to_datetime(df_ug['dateTimeMeasured'], format='%Y-%m-%d %H:%M:%S')
df_ug = df_ug.sort_values(by=["dateTimeMeasured"], ascending=True, ignore_index=True)

# Make column names consistent
df_ug['cell_temperature'] = celsius_to_kelvin(df_ug['TCellRef'])
df_ug['cell_temperature_error'] = df_ug['TCellSam_error']
df_ug['cell_pressure'] = torr_to_pascal(df_ug['PCellSam'])
df_ug['cell_pressure_error'] = torr_to_pascal(df_ug['PCellSam_error'])
df_ug['chi_p_626'] = df_ug['pCO2Ref']
df_ug['chi_p_626_error'] = df_ug['pCO2Ref_error']
df_ug['d18O'] = deltaO_vs_WG(df_ug['d18O'], 28.048)
df_ug['d17O'] = deltaO_vs_WG(df_ug['d17O'], 14.621)
df_ug['Dp17O'] = Dp17O(df_ug['d17O'], df_ug['d18O'])
df_ug['Dp17O_error'] = df_ug['Dp17OError']

separator = ["~"*40, "~"*40, "~"*40]
table = [
    ["Pressure range across replicates",
     f"{df_ug['cell_pressure'].min():.0f} to {df_ug['cell_pressure'].max():.0f} Pa (mean: {df_ug['cell_pressure'].mean():.0f})",
     f"{df_uct['cell_pressure'].min():.0f} to {df_uct['cell_pressure'].max():.0f} Pa (mean: {df_uct['cell_pressure'].mean():.0f})"],

    ["",
     f"{pascal_to_torr(df_ug['cell_pressure']).min():.1f} to {pascal_to_torr(df_ug['cell_pressure']).max():.1f} Torr (mean: {pascal_to_torr(df_ug['cell_pressure']).mean():.1f})",
     f"{pascal_to_torr(df_uct['cell_pressure']).min():.1f} to {pascal_to_torr(df_uct['cell_pressure']).max():.1f} Torr (mean: {pascal_to_torr(df_uct['cell_pressure']).mean():.1f})"],

    ["Pressure stability within replicate",
     f"±{df_ug['cell_pressure_error'].mean():.1f} Pa",
     f"±{df_uct['cell_pressure_error'].mean():.1f} Pa"],

    ["",
     f"±{pascal_to_torr(df_ug['cell_pressure_error'].mean()*1000):.0f} mTorr",
     f"±{pascal_to_torr(df_uct['cell_pressure_error'].mean()*1000):.0f} mTorr"],

    ["Pressure mismatch across replicates",
     f"{torr_to_pascal(df_ug['PCellMismatch'].min()):.1f} to {torr_to_pascal(df_ug['PCellMismatch'].max()):.1f} Pa",
     f"{torr_to_pascal(df_uct['mismatch_Praw'].min()):.1f} to {torr_to_pascal(df_uct['mismatch_Praw'].max()):.1f} Pa"],

    ["",
     f"{(df_ug['PCellMismatch'].min()*1000):.0f} to {(df_ug['PCellMismatch'].max()*1000):.0f} mTorr",
     f"{(df_uct['mismatch_Praw'].min()*1000):.0f} to {(df_uct['mismatch_Praw'].max()*1000):.0f} mTorr"],


    ["Pressure mismatch stability",
     f"±{torr_to_pascal(df_ug['PCellMismatch_error'].mean()):.1f} Pa",
     f"±{torr_to_pascal(df_uct['std_mismatch_Praw'].mean()):.1f} Pa"],

    ["",
     f"±{(df_ug['PCellMismatch_error'].mean()*1000):.0f} mTorr",
     f"±{(df_uct['std_mismatch_Praw'].mean()*1000):.0f} mTorr"],

    separator,

    ["Temperature range across replicates",
     f"{df_ug['cell_temperature'].min():.1f} to {df_ug['cell_temperature'].max():.1f} K (mean: {df_ug['cell_temperature'].mean():.1f} K)",
     f"{df_uct['cell_temperature'].min():.1f} to {df_uct['cell_temperature'].max():.1f} K (mean: {df_uct['cell_temperature'].mean():.1f} K)"],

    ["Temperature stability within replicate",
     f"±{df_ug['cell_temperature_error'].mean()*1000:.0f} mK",
     f"±{df_uct['cell_temperature_error'].mean()*1000:.0f} mK"],

    ["Temperature mismatch across replicates",
     f"{df_ug['TCellMismatch'].min()*1000:.0f} to {df_ug['TCellMismatch'].max()*1000:.0f} mK",
     f"{df_uct['mismatch_Traw'].min()*1000:.0f} to {df_uct['mismatch_Traw'].max()*1000:.0f} mK"],

    ["Temperature mismatch stability",
     f"±{df_ug['TCellMismatch_error'].mean()*1000:.0f} mK",
     f"±{df_uct['std_mismatch_Traw'].mean()*1000:.0f} mK"],

    separator,

    ["Xp626 range across replicates",
     f"{df_ug['chi_p_626'].min():.0f} to {df_ug['chi_p_626'].max():.0f} µmol/mol",
     f"{df_uct['chi_p_626'].min():.0f} to {df_uct['chi_p_626'].max():.0f} µmol/mol"],

    ["Xp626 stability within replicate",
     f"±{df_ug['chi_p_626_error'].mean():.1f} µmol/mol",
     f"±{df_uct['chi_p_626_error'].mean():.1f} µmol/mol"],

    ["Xp626 mismatch across replicates",
     f"{df_ug['pCO2Mismatch'].min():.1f} to {df_ug['pCO2Mismatch'].max():.1f} µmol/mol",
     f"{df_uct['mismatch_Xp626_L2'].min()/1000:.1f} to {df_uct['mismatch_Xp626_L2'].max()/1000:.1f} µmol/mol"],

    ["Xp626 mismatch stability",
     f"±{df_ug['pCO2Mismatch_error'].mean():.1f} µmol/mol",
     f"±{df_uct['std_mismatch_Xp626_L2'].mean()/1000:.1f} µmol/mol"],
]
print(tabulate(table, headers=["Parameter", "UG", "UCT"], tablefmt="grid"))


# scatterplot of devmean chi_p_626 vs devmean Dp17O for each unique sample
fig, axes = plt.subplots(nrows=3, ncols=2, sharex='col')
axes = axes.flatten()

for ax in fig.get_axes():
    i = fig.get_axes().index(ax)
    ax.text(0.025, 0.975, chr(97 + i),
            size=12, weight="bold", ha="left", va="top",
            transform=ax.transAxes)


def add_linfit_and_stats(ax, x, y):
    # Linear fit
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    xx = np.array([np.nanmin(x), np.nanmax(x)])
    yy = slope * xx + intercept
    ax.plot(xx, yy, ls="--", c="k")
    R2_val = r_value**2
    if R2_val >= 0.1:
        R2_str = r"$R^2$ = " + f"{R2_val:.1g}"
    else:
        R2_str = r"$R^2 < 0.1$"

    if p_value < 0.05:
        p_str = r"$\it{p} < 0.05$"
    else:
        p_str = r"$\it{p}$ = " + f"{p_value:.1g}"

    txt = f"{R2_str}\n{p_str}"
    ax.text(0.77, 0.98, txt, transform=ax.transAxes,
            va='top', ha='left', color="k",
            bbox=dict(fc='white', alpha=0.8, ec='none', pad=0.2, boxstyle='round'))


def add_linfit_and_stats_95(ax, x, y):

    # Filter values to 95% in y
    mask = y <= np.nanpercentile(y, 90)
    x = x[mask]
    y = y[mask]

    # Linear fit
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    xx = np.array([np.nanmin(x), np.nanmax(x)])
    yy = slope * xx + intercept
    ax.plot(xx, yy, ls="--", c="r")
    R2_val = r_value**2
    if R2_val >= 0.1:
        R2_str = r"$R^2$ = " + f"{R2_val:.1g}"
    else:
        R2_str = r"$R^2 < 0.1$"

    if p_value < 0.05:
        p_str = r"$\it{p} < 0.05$"
    else:
        p_str = r"$\it{p}$ = " + f"{p_value:.1g}"

    txt = f"{R2_str}\n{p_str}"
    ax.text(0.45, 0.98, txt, transform=ax.transAxes,
            va='top', ha='left', color="r",
            bbox=dict(fc='white', alpha=0.8, ec='none', pad=0.2, boxstyle='round'))
    
    print(f"90% filtered data maximum value: {y.max():.1f}")



# Left column - Göttingen
axes[0].scatter(df_ug['Dp17O_error'], df_ug['pCO2Mismatch_error'], 
                facecolor="#1455C080", ec="#1455C0")
add_linfit_and_stats(axes[0], df_ug['Dp17O_error'], df_ug['pCO2Mismatch_error'])
print(f"UG, maximum x626 mismatch stability: {df_ug['pCO2Mismatch_error'].max():.1f} µmol/mol\n")

axes[2].scatter(df_ug['Dp17O_error'], torr_to_pascal(df_ug['PCellMismatch_error']), 
                facecolor="#1455C080", ec="#1455C0")
add_linfit_and_stats(axes[2], df_ug['Dp17O_error'], torr_to_pascal(df_ug['PCellMismatch_error']))
print(f"UG, maximum pressure mismatch stability: {torr_to_pascal(df_ug['PCellMismatch_error']).max():.1f} Pa, {df_ug['PCellMismatch_error'].max()*1000:.1f} mTorr\n")

axes[4].scatter(df_ug['Dp17O_error'], df_ug['TCellMismatch_error']*1000, 
                facecolor="#1455C080", ec="#1455C0")
add_linfit_and_stats(axes[4], df_ug['Dp17O_error'], df_ug['TCellMismatch_error']*1000)
print(f"UG, maximum temperature mismatch stability: {df_ug['TCellMismatch_error'].max()*1000:.0f} mK\n")

# Right column - Cape Town
axes[1].scatter(df_uct['Dp17O_error'], df_uct['std_mismatch_Xp626_L2']/1000, 
                fc="#81499780", ec="#814997")
add_linfit_and_stats(axes[1], df_uct['Dp17O_error'], df_uct['std_mismatch_Xp626_L2']/1000)
add_linfit_and_stats_95(axes[1], df_uct['Dp17O_error'], df_uct['std_mismatch_Xp626_L2']/1000)
print(f"UCT, maximum x626 mismatch stability: {df_uct['std_mismatch_Xp626_L2'].max()/1000:.1f} µmol/mol\n")

axes[3].scatter(df_uct['Dp17O_error'], torr_to_pascal(df_uct['std_mismatch_Praw']), 
                fc="#81499780", ec="#814997")
add_linfit_and_stats(axes[3], df_uct['Dp17O_error'], torr_to_pascal(df_uct['std_mismatch_Praw']))
add_linfit_and_stats_95(axes[3], df_uct['Dp17O_error'], torr_to_pascal(df_uct['std_mismatch_Praw']))
print(f"UCT, maximum pressure mismatch stability: {torr_to_pascal(df_uct['std_mismatch_Praw']).max():.1f} Pa, {df_uct['std_mismatch_Praw'].max()*1000:.1f} mTorr\n")

axes[5].scatter(df_uct['Dp17O_error'], df_uct['std_mismatch_Traw']*1000, 
                fc="#81499780", ec="#814997")
add_linfit_and_stats(axes[5], df_uct['Dp17O_error'], df_uct['std_mismatch_Traw']*1000)
add_linfit_and_stats_95(axes[5], df_uct['Dp17O_error'], df_uct['std_mismatch_Traw']*1000)
print(f"UCT, maximum temperature mismatch stability: {df_uct['std_mismatch_Traw'].max()*1000:.0f} mK\n")

# ax.set_xlabel("devmean_chi_p_626")
axes[4].set_xlabel("Internal $\Delta\prime^{17}O$ error (ppm)")
axes[5].set_xlabel("Internal $\Delta\prime^{17}O$ error (ppm)")

axes[0].set_ylabel("Stability of\n$\chi\prime_{626}$ mismatch (µmol/mol)")
axes[2].set_ylabel("Stability of\nPressure mismatch (Pa)")
axes[4].set_ylabel("Stability of\ntemperature mismatch (mK)")
axes[1].set_ylabel("Stability of\n$\chi\prime_{626}$ mismatch (µmol/mol)")
axes[3].set_ylabel("Stability of\nPressure mismatch (Pa)")
axes[5].set_ylabel("Stability of\ntemperature mismatch (mK)")

axes[0].set_title("University of Göttingen")
axes[1].set_title("University of Cape Town")

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "LT_Figure_6.png"))
plt.close("all")


# new plot one rwo two columns
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
axes = axes.flatten()

for ax in fig.get_axes():
    i = fig.get_axes().index(ax)
    ax.text(0.025, 0.975, chr(97 + i),
            size=12, weight="bold", ha="left", va="top",
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=0.2, boxstyle='round'),
            transform=ax.transAxes)

df_ug['Dp17O_centered'] = df_ug['Dp17O'] - df_ug.groupby('SampleName')['Dp17O'].transform('mean')
axes[0].scatter(df_ug['pCO2Mismatch'], df_ug['Dp17O_centered'], 
                fc="#1455C080", ec="#1455C0")
add_linfit_and_stats(axes[0], df_ug['pCO2Mismatch'], df_ug['Dp17O_centered'])


df_uct['Dp17O_centered'] = df_uct['Dp17O_meas'] - df_uct.groupby('SampleName')['Dp17O_meas'].transform('mean')
axes[1].scatter(df_uct['mismatch_Xp626_L2']/1000, df_uct['Dp17O_centered'], 
                fc="#81499780", ec="#814997")
add_linfit_and_stats(axes[1], df_uct['mismatch_Xp626_L2']/1000, df_uct['Dp17O_centered'])

axes[0].set_ylabel("$\Delta\prime^{17}O_{smp/std}$ (ppm, mean-centered)")
axes[1].set_ylabel("$\Delta\prime^{17}O_{smp/std}$ (ppm, mean-centered)")

axes[0].set_xlabel(r"$\chi\prime^{sam}_{626} - \chi\prime^{std}_{626} $ ($\mu$mol mol$^{-1}$)")
axes[1].set_xlabel(r"$\chi\prime^{sam}_{626} - \chi\prime^{std}_{626} $ ($\mu$mol mol$^{-1}$)")

axes[0].set_title("University of Göttingen")
axes[1].set_title("University of Cape Town")

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "LT_Figure_2.png"))
plt.close("all")