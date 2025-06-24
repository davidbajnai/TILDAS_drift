"""
This script shows the effect of scale offset correction on the Cape Town data.

INPUT:
- LT_Table_S2.csv: A CSV file containing data from the University of Cape Town.

OUTPUT:
- LT_Figure_4.png: A plot showing the Dp17O values before and after scale offset correction.
"""


# Import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from functions import *

# Get script directory for relative file paths
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get data directory
data_dir = os.path.join(script_dir, '../data')

# Get figures directory
figures_dir = os.path.join(script_dir, '../figures')

# Plot parameters
plt.rcParams.update({"font.size": 8})
plt.rcParams["figure.figsize"] = (6, 6)
plt.rcParams["patch.linewidth"] = 0.5
plt.rcParams['lines.linewidth'] = 0.5
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams['savefig.transparent'] = False
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['legend.loc'] = 'upper right'

# Colors for plotting the reference materials
c_NBS18, c_IAEA603 = "#F75056", "#4D0820"
c_light, c_heavy = "#309FD1", "#0C3992"

# Get data - University of Cape Town
df_uct = pd.read_csv(os.path.join(data_dir, "LT_Table_S2.csv"))
df_uct['dateTimeMeasured'] = pd.to_datetime(df_uct['Name'], format='%y%m%d_%H%M%S', errors='coerce')
df_uct['dateTimeMeasured_num'] = df_uct["dateTimeMeasured"].astype(np.int64) // 10**9
df_uct = df_uct.sort_values(by=["dateTimeMeasured"], ascending=True, ignore_index=True)

# Make column names consistent for plotting
df_uct["SampleName"] = df_uct['AnalyticalID']
df_uct['Dp17O'] = df_uct['Dp17O_raw_mean']*1000
df_uct['Dp17O_corr'] = df_uct['Dp17O_corr']*1000
df_uct['Dp17O_error'] = df_uct['Dp17O_raw_std']*1000
df_uct['cell_temperature'] = df_uct['Traw_mean']
df_uct['cell_temperature_error'] = df_uct['Traw_std']
df_uct['cell_pressure'] = torr_to_pascal(df_uct['Praw_mean'])
df_uct['cell_pressure_error'] = df_uct['Praw_std']*133.322
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
df_ug['dateTimeMeasured_num'] = df_ug["dateTimeMeasured"].astype(np.int64) // 10**9

# Make column names consistent for plotting
df_ug['Dp17O_error'] = df_ug['Dp17OError']
df_ug['cell_temperature'] = celsius_to_kelvin(df_ug['TCellRef'])
df_ug['cell_temperature_error'] = df_ug['TCellSam_error']
df_ug['cell_pressure'] = torr_to_pascal(df_ug['PCellRef'])
df_ug['cell_pressure_error'] = df_ug['PCellRef_error']*133.322
df_ug['chi_p_626'] = df_ug['pCO2Sam'] # in ppmv
df_ug['chi_p_626_error'] = df_ug['pCO2Ref_error']

# Convert d-values relative to WG (so a zero enrichment would yield a d18O of 0)
df_ug["d18O"] = deltaO_vs_WG(df_ug["d18O"], 28.048)
df_ug["d17O"] = deltaO_vs_WG(df_ug["d17O"], 14.621)
df_ug["Dp17O"] = Dp17O(df_ug["d17O"], df_ug["d18O"])
df_ug["Dp17O_corr"] = Dp17O(df_ug["d17O"], df_ug["d18O"]) + (df_ug["pCO2Ref"]-416) * 6


# Function for plotting
def plot_data(df, lab_ref1, c_ref1, lab_ref2, c_ref2, fig_num):

    df = df[df['SampleName'].str.contains(lab_ref1, regex=False) | df['SampleName'].str.contains(lab_ref2, regex=False)]
    common_x = np.linspace(df["dateTimeMeasured_num"].min(), df["dateTimeMeasured_num"].max(), 10000)

    # Apply LOESS smoothing and store results
    df_fit = pd.DataFrame({"common_x": common_x})

    df_fit["Dp17O_ref1"] = apply_loess(df, "Dp17O", common_x, lab_ref1)["loess_fit"]
    df_fit["Dp17O_ref2"] = apply_loess(df, "Dp17O", common_x, lab_ref2)["loess_fit"]
    df_fit["d18O_ref1"] = apply_loess(df, "d18O", common_x, lab_ref1)["loess_fit"]
    df_fit["d18O_ref2"] = apply_loess(df, "d18O", common_x, lab_ref2)["loess_fit"]
    df_fit["d17O_ref1"] = apply_loess(df, "d17O", common_x, lab_ref1)["loess_fit"]
    df_fit["d17O_ref2"] = apply_loess(df, "d17O", common_x, lab_ref2)["loess_fit"]
    df_fit["chi_p_626"] = apply_loess(df, "chi_p_626", common_x)["loess_fit"]

    df_ref1 = df[df['SampleName'].str.contains(lab_ref1, regex=False)]
    df_ref2 = df[df['SampleName'].str.contains(lab_ref2, regex=False)]

    fig, axes = plt.subplots(3, 1, sharex=True)
    axes = axes.flatten()

    for ax in fig.get_axes():
        i = fig.get_axes().index(ax)
        ax.text(0.01, 0.975, chr(97 + i),
                size=12, weight="bold", ha="left", va="top",
                transform=ax.transAxes)

    # Subplot A: ∆'17O data and LOESS fit
    axes[0].errorbar(df_ref1['dateTimeMeasured'],
                 (df_ref1['Dp17O']),
                 yerr=df_ref1['Dp17O_error'],
                 fmt='o',
                 mec=c_ref1,
                 mew = 0.5,
                 ms=5,
                 c=f"{c_ref1}80",
                 label=lab_ref1)
    axes[0].errorbar(df_ref2['dateTimeMeasured'],
                 (df_ref2['Dp17O']),
                 yerr=df_ref2['Dp17O_error'],
                 fmt='o',
                 mec=c_ref2,
                 mew=0.5,
                 ms=5,
                 c=f"{c_ref2}80",
                 label=lab_ref2)
    
    print(f"1SD of {lab_ref1} Dp17O: {df_ref1['Dp17O'].std():.0f} ppm")
    print(f"1SD of {lab_ref2} Dp17O: {df_ref2['Dp17O'].std():.0f} ppm")

    axes[0].legend(loc="upper right")
    axes[0].set_ylabel("$\Delta\prime^{17}O_{smp/std}$ (ppm)")
    axes[0].set_ylim(-420, 420)


    # Subplot B: Mixing ratio
    axes[2].errorbar(df_ref1['dateTimeMeasured'],
                 df_ref1['chi_p_626'],
                 yerr=df_ref1['chi_p_626_error'],
                 fmt='o',
                 mec=c_ref1,
                 mew=0.5,
                 ms=5,
                 c=f"{c_ref1}80",
                 label=lab_ref1)
    axes[2].errorbar(df_ref2['dateTimeMeasured'],
                 df_ref2['chi_p_626'],
                 yerr=df_ref2['chi_p_626_error'],
                 fmt='o',
                 mec=c_ref2,
                 mew=0.5,
                 ms=5,
                 c=f"{c_ref2}80",
                 label=lab_ref2)

    axes[2].axhline(421, ls='--', c='k', lw=1, label="Reference value (421 ppmv)", zorder = -1)

    axes[2].set_ylabel('$\chi\prime^{sam}_{626}$ (µmol mol$^{-1}$)')


    # Subplot A: ∆'17O data and LOESS fit
    axes[1].errorbar(df_ref1['dateTimeMeasured'],
                 (df_ref1['Dp17O_corr']),
                 yerr=df_ref1['Dp17O_error'],
                 fmt='o',
                 mec=c_ref1,
                 mew = 0.5,
                 ms=5,
                 c=f"{c_ref1}80",
                 label=lab_ref1)
    axes[1].errorbar(df_ref2['dateTimeMeasured'],
                 (df_ref2['Dp17O_corr']),
                 yerr=df_ref2['Dp17O_error'],
                 fmt='o',
                 mec=c_ref2,
                 mew=0.5,
                 ms=5,
                 c=f"{c_ref2}80",
                 label=lab_ref2)
    
    print(f"1SD of {lab_ref1} Dp17O_corr: {df_ref1['Dp17O_corr'].std():.0f} ppm")
    print(f"1SD of {lab_ref2} Dp17O_corr: {df_ref2['Dp17O_corr'].std():.0f} ppm")


    axes[1].set_ylabel("$\Delta\prime^{17}O^{true}_{smp/std}$ (ppm)")
    axes[1].set_ylim(-420, 420)

    axes[-1].set_xlabel('Measurement date')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"LT_Figure_{fig_num}.png"))
    plt.close("all")


# plot_data(df_ug, "light $CO_2$", c_light, "heavy $CO_2$", c_heavy, 99)
plot_data(df_uct, "NBS18", c_NBS18, "IAEA603", c_IAEA603, 4)