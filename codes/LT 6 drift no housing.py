"""
This script plots analytical conditions and isotopic data from a pair of STR and STC files.

INPUT:
- 220313_094743.str: A STR file containing isotopic data.
- 220313_094743.stc: A STC file containing analytical conditions data.

OUTPUT:
- LT Figure 5.png: A plot showing the isotopic data and analytical conditions over time.
"""

# Import libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from functions import *

# Get script directory for relative file paths
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get data directory
data_dir = os.path.join(script_dir, '../data')

# Get figures directory
figures_dir = os.path.join(script_dir, '../figures')

# Plot parameters
plt.rcParams.update({"font.size": 8})
plt.rcParams["figure.figsize"] = (5, 6)
plt.rcParams["patch.linewidth"] = 1
plt.rcParams['lines.linewidth'] = 1
plt.rcParams["savefig.dpi"] = 600
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams['savefig.transparent'] = False
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['legend.loc'] = 'lower right'

# Import data
filename = "220313_094743"
dfSTC = pd.read_csv(os.path.join(data_dir, f"{filename}.stc"), skiprows=1)
dfSTC.columns = dfSTC.columns.str.strip()
colnames = ["Time(abs)", "627", "628", "626", "CO2"]
dfSTR = pd.read_csv(os.path.join(data_dir, f"{filename}.str"), skiprows=1, names=colnames, header=None, sep='\s+')
df = pd.concat([dfSTR, dfSTC], axis=1, join="inner")

# Calculate relative time in seconds from the first time point
rel_time = df["Time(abs)"] - df["Time(abs)"].iloc[0]

# Create a mask to exclude the first and last 300 seconds
mask = (rel_time >= 300) & (rel_time <= rel_time.max() - 300)
df = df[mask].copy()
df.set_index(rel_time[mask] / 3600, inplace=True)
df.index.name = "Relative time (h)"

# Calculate isotope ratios
df["d18O"] = ((df["628"]/df["626"])-1) * 1000
df["d17O"] = ((df["627"]/df["626"])-1) * 1000
df["Dp17O"] = Dp17O(df["d17O"], df["d18O"])

fig, axes = plt.subplots(4, 1, sharex=True)
axes = axes.flatten()

for ax in fig.get_axes():
    i = fig.get_axes().index(ax)
    ax.text(0.015, 0.975, chr(97 + i),
            size=12, weight="bold", ha="left", va="top",
            bbox=dict(fc="w", pad=0.1, ec="none", alpha=0.8),
            transform=ax.transAxes)

# Cell temperature
axes[0].plot(df.index, devmean(df["Traw"]), c="#1455C0")
axes[0].set_ylabel(r'$\mathit{T}$ (K)')


# Concentration
y_628 = devmean(mole_fraction_to_concentration(df["628"]/10**9, df["Traw"], torr_to_pascal(df["Praw"]), 628))*10**8
axes[1].plot(df.index, y_628, c="#814997", label="'628' (x10$^2$)")

y_626 = devmean(mole_fraction_to_concentration(df["626"]/10**9, df["Traw"], torr_to_pascal(df["Praw"]), 626))*10**6
axes[1].plot(df.index, y_626, c="#F75056", label="'626'")

y_627 = devmean(mole_fraction_to_concentration(df["627"]/10**9, df["Traw"], df["Praw"]*133.322, 627))*10**9
axes[1].plot(df.index, y_627, c="#1455C0", label="'627' (x10$^3$)")

axes[1].legend()
axes[1].set_ylabel("$\mathit{C}$ (µmol m$^{-3}$)")


# Delta values
axes[2].plot(df.index, devmean(df["d18O"]), c="#814997", label="$\delta^{18}O$")
axes[2].plot(df.index, devmean(df["d17O"]), c="#1455C0", label="$\delta^{17}O$")

axes[2].legend()
axes[2].set_ylabel("$\delta_{meas}$ (‰)")

# Dp17O
axes[3].plot(df.index, devmean(df["Dp17O"]), c="#1455C0")
axes[3].set_ylabel("$\Delta\prime^{17}O_{meas}$ (ppm)")


axes[-1].xaxis.set_major_locator(MultipleLocator(1))
axes[-1].set_xlabel("Measurement time (h)")

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "LT Figure 5.png"))
plt.close("all")