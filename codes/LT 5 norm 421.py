"""
This script shows the effect of scale offset correction on the Cape Town data.

INPUT:
- LT_Table_S2.csv: A CSV file containing data from the University of Cape Town.

OUTPUT:
- LT_Figure_4.png: A plot showing the ∆'17O values before and after scale-offset correction.
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
c_NBS18, c_IAEA603 = "#F75056", "#814997"

# Get data - University of Cape Town
df_uct = pd.read_csv(os.path.join(data_dir, "LT_Table_S2.csv"))
df_uct['dateTimeMeasured'] = pd.to_datetime(df_uct['dateTimeMeasured'], format='%Y-%m-%d %H:%M:%S')
df_uct['dateTimeMeasured_num'] = df_uct["dateTimeMeasured"].astype(np.int64) // 10**9
df_uct = df_uct.sort_values(by=["dateTimeMeasured"], ascending=True, ignore_index=True)

# Make column names consistent for plotting
df_uct["SampleName"] = df_uct['ID']
df_uct['Dp17O'] = df_uct['Dp17O_raw_mean']*1000
df_uct['Dp17O_corr'] = df_uct['Dp17O_corr']*1000
df_uct['Dp17O_error'] = df_uct['Dp17O_raw_std']*1000
df_uct['cell_temperature'] = df_uct['Traw_mean']
df_uct['cell_temperature_error'] = df_uct['Traw_std']
df_uct['electronics_temperature'] = df_uct['AD8']
df_uct['electronics_temperature_error'] = df_uct['AD8_std']
df_uct['cell_pressure'] = torr_to_pascal(df_uct['Praw_mean'])
df_uct['cell_pressure_error'] = df_uct['Praw_std']*133.322
df_uct['chi_p_626'] = df_uct['Xp626_L2_mean']/1000
df_uct['chi_p_626_error'] = df_uct['Xp626_L2_std']/1000
df_uct['d18O'] = unprime(df_uct['dp18O_L2_raw_mean'])
df_uct['d17O'] = unprime(df_uct['dp17O_L2_raw_mean'])

# Exclude some samples based on ID
df_uct = df_uct[df_uct['ID'] != "IAEA603-27"]
df_uct = df_uct[df_uct['ID'] != "IAEA603-33"]
df_uct = df_uct[df_uct['ID'] != "IAEA603-35"]
df_uct = df_uct[df_uct['ID'] != "IAEA603-36"]
df_uct = df_uct[df_uct['ID'] != "IAEA603-59"]
df_uct = df_uct[df_uct['ID'] != "NBS18-36"]
df_uct = df_uct[df_uct['ID'] != "NBS18-37"]
df_uct = df_uct[df_uct['ID'] != "NBS18-56"]
df_uct = df_uct[df_uct['ID'] != "NBS18-57"]

# Placing the plotting in a function makes it easier to apply it for different datasets
def plot_data(df, lab_ref1, c_ref1, lab_ref2, c_ref2, fig_num):

    df = df[df['SampleName'].str.contains(lab_ref1, regex=False) | df['SampleName'].str.contains(lab_ref2, regex=False)]

    df_ref1 = df[df['SampleName'].str.contains(lab_ref1, regex=False)]
    df_ref2 = df[df['SampleName'].str.contains(lab_ref2, regex=False)]

    fig, axes = plt.subplots(3, 1, sharex=True)
    axes = axes.flatten()

    for ax in fig.get_axes():
        i = fig.get_axes().index(ax)
        ax.text(0.01, 0.975, chr(97 + i),
                size=12, weight="bold", ha="left", va="top",
                transform=ax.transAxes)

    # Subplot A: measured ∆'17O 
    axes[0].errorbar(df_ref1['dateTimeMeasured'],
                     (df_ref1['Dp17O']),
                     yerr=df_ref1['Dp17O_error'],
                     fmt='o',
                     mec=c_ref1,
                     mew=0.5,
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
    
    axes[0].legend(loc="upper right")
    axes[0].set_ylabel("$\Delta\prime^{17}O^{smp/wg}_{meas}$ (ppm)")
    axes[0].set_ylim(-420, 420)


    # Subplot B: scale-offset corrected, "true" ∆'17O
    axes[1].errorbar(df_ref1['dateTimeMeasured'],
                     (df_ref1['Dp17O_corr']),
                     yerr=df_ref1['Dp17O_error'],
                     fmt='o',
                     mec=c_ref1,
                     mew=0.5,
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

    axes[1].set_ylabel("$\Delta\prime^{17}O^{smp/wg}_{true}$ (ppm)")
    axes[1].set_ylim(-420, 420)


    # Subplot C: Mixing ratio
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

    axes[2].axhline(421,
                    ls='--', c='k', lw=1, zorder=-1,
                    label="Reference value (421 ppmv)")

    axes[2].set_ylabel('$\chi\prime^{smp}_{626, meas}$ (µmol mol$^{-1}$)')


    axes[-1].set_xlabel('Measurement date')
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))

    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f"LT_Figure_{fig_num}.png"))
    plt.close("all")

    print("Comparing the standard deviation of standards before and after scale-offset correction:")
    print(f"{lab_ref1} ∆'17O_meas: {df_ref1['Dp17O'].std():.0f} ppm ---> ∆'17O_true: {df_ref1['Dp17O_corr'].std():.0f} ppm")
    print(f"{lab_ref2} ∆'17O_meas: {df_ref2['Dp17O'].std():.0f} ppm ---> ∆'17O_true: {df_ref2['Dp17O_corr'].std():.0f} ppm")

plot_data(df_uct, "NBS18", c_NBS18, "IAEA603", c_IAEA603, 4)
