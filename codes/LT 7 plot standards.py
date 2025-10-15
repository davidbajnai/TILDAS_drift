"""
This script illustrates the long-term drift of the analytical parameters
and its impact on the measured ∆'17O values.

INPUT:
- LT_Table_S1.csv: A CSV file containing data from the University of Göttingen.
- LT_Table_S2.csv: A CSV file containing data from the University of Cape Town.

OUTPUT:
- LT_Figure_7.png: Long-term drift of ∆'17O measurements at the University of Göttingen.
- LT_Figure_8.png: Long-term drift of ∆'17O measurements at the University of Cape Town.
- LT_Figure_9.png: Correlation coefficients from the multivariate linear regression models.
"""

# Import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from statsmodels.api import OLS, add_constant
from functions import *

# Get script directory for relative file paths
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get data directory
data_dir = os.path.join(script_dir, '../data')

# Get figures directory
figures_dir = os.path.join(script_dir, '../figures')

# Plot parameters
plt.rcParams.update({"font.size": 8})
plt.rcParams["patch.linewidth"] = 0.5
plt.rcParams['lines.linewidth'] = 0.5
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams['savefig.transparent'] = False
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['legend.loc'] = 'upper right'

# Colors for plotting the reference materials
c_NBS18, c_IAEA603 = "#F75056", "#814997"
c_light, c_heavy = "#309FD1", "#0C3992"

# Get data - University of Cape Town
df_uct = pd.read_csv(os.path.join(data_dir, "LT_Table_S2.csv"))
df_uct['dateTimeMeasured'] = pd.to_datetime(df_uct['dateTimeMeasured'], format='%Y-%m-%d %H:%M:%S')
df_uct['dateTimeMeasured_num'] = df_uct["dateTimeMeasured"].astype(np.int64) // 10**9
df_uct = df_uct.sort_values(by=["dateTimeMeasured"], ascending=True, ignore_index=True)

# Make column names consistent for plotting
df_uct["SampleName"] = df_uct['ID']
df_uct['Dp17O'] = df_uct['Dp17O_corr']*1000
df_uct['Dp17O_error'] = df_uct['Dp17O_raw_std']*1000
df_uct['cell_temperature'] = df_uct['Traw_mean']
df_uct['cell_temperature_error'] = df_uct['Traw_std']
df_uct['electronics_temperature'] = df_uct['AD8']
df_uct['electronics_temperature_error'] = df_uct['AD8_std']
df_uct['cell_pressure'] = torr_to_pascal(df_uct['Praw_mean'])
df_uct['cell_pressure_error'] = df_uct['Praw_std']*133.322
df_uct['chi_p_626'] = 421
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

# Get data - University of Göttingen
df_ug = pd.read_csv(os.path.join(data_dir, "LT_Table_S1.csv"))
df_ug['SampleName'] = df_ug['SampleName'].str.replace("VsRef", " $CO_2$")
df_ug['dateTimeMeasured'] = pd.to_datetime(df_ug['dateTimeMeasured'], format='%Y-%m-%d %H:%M:%S')
df_ug = df_ug.sort_values(by=["dateTimeMeasured"], ascending=True, ignore_index=True)
df_ug['dateTimeMeasured_num'] = df_ug["dateTimeMeasured"].astype(np.int64) // 10**9

# Make column names consistent for plotting
df_ug['Dp17O_error'] = df_ug['Dp17OError']
df_ug['cell_temperature'] = celsius_to_kelvin(df_ug['TCellSam'])
df_ug['cell_temperature_error'] = df_ug['TCellSam_error']
df_ug['electronics_temperature'] = celsius_to_kelvin(df_ug['TIntSam'])
df_ug['electronics_temperature_error'] = df_ug['TIntSam_error']
df_ug['cell_pressure'] = torr_to_pascal(df_ug['PCellRef'])
df_ug['cell_pressure_error'] = df_ug['PCellRef_error']*133.322
df_ug['chi_p_626'] = df_ug['pCO2Ref']
df_ug['chi_p_626_error'] = df_ug['pCO2Ref_error']

# Rolling standard deviation over a 7-day window (centered)
df_ug.index = pd.to_datetime(df_ug.index)
rolling_scatter = df_ug["chi_p_626"].rolling("7D").std()
rolling_scatter = rolling_scatter.dropna()
print(f"Average change in pCO2 between measurements at the University of Göttingen: {rolling_scatter.mean():.0f} ppm")

# Function to apply LOESS smoothing
def loess_on_columns(df, lab_ref1, lab_ref2):

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
    df_fit["cell_pressure"] = apply_loess(df, "cell_pressure", common_x)["loess_fit"]
    df_fit["cell_temperature"] = apply_loess(df, "cell_temperature", common_x)["loess_fit"]
    df_fit["electronics_temperature"] = apply_loess(df, "electronics_temperature", common_x)["loess_fit"]
    df_fit["chi_p_626"] = apply_loess(df, "chi_p_626", common_x)["loess_fit"]

    df_fit["Dp17O_compression"] = (df_fit["Dp17O_ref1"] - df_fit["Dp17O_ref2"])

    df_ug_ref1 = df[df['SampleName'].str.contains(lab_ref1, regex=False)]
    df_ref2 = df[df['SampleName'].str.contains(lab_ref2, regex=False)]

    return df_fit, df_ug_ref1, df_ref2, lab_ref1, lab_ref2



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Figure 7 - University of Göttingen

# Apply LOESS smoothing for University of Göttingen data
df_ug_fit, df_ug_ref1, df_ug_ref2, lab_ref1_ug, lab_ref2_ug = loess_on_columns(df_ug, "light $CO_2$", "heavy $CO_2$")

fig, axes = plt.subplots(6, 1, sharex=True, figsize=(6, 8.5))
axes = axes.flatten()

for ax in fig.get_axes():
    i = fig.get_axes().index(ax)
    ax.text(0.01, 0.975, chr(97 + i),
            size=12, weight="bold", ha="left", va="top",
            transform=ax.transAxes)

# Subplot -  ∆'17O data
axes[0].errorbar(df_ug_ref1['dateTimeMeasured'],
                 (df_ug_ref1['Dp17O']),
                 yerr=df_ug_ref1['Dp17O_error'],
                 fmt='o',
                 mec=c_light,
                 mew=0.5,
                 ms=5,
                 c=f"{c_light}80",
                 label=lab_ref1_ug)
axes[0].errorbar(df_ug_ref2['dateTimeMeasured'],
                 (df_ug_ref2['Dp17O']),
                 yerr=df_ug_ref2['Dp17O_error'],
                 fmt='o',
                 mec=c_heavy,
                 mew=0.5,
                 ms=5,
                 c=f"{c_heavy}80",
                 label=lab_ref2_ug)

axes[0].plot(pd.to_datetime(df_ug_fit['common_x'], unit='s'), (df_ug_fit['Dp17O_ref1']),
             '-', c=c_light, lw=2)
axes[0].plot(pd.to_datetime(df_ug_fit['common_x'], unit='s'), (df_ug_fit['Dp17O_ref2']),
             '-', c=c_heavy, lw=2)

axes[0].plot([], [], '-', c="k", lw=2, label="LOESS fit")
axes[0].legend(loc="upper right")

axes[0].set_ylabel("$\Delta\prime^{17}O^{smp/wg}_{meas}$ (ppm)")

# Subplot – 626 mixing ratio
axes[1].errorbar(df_ug_ref1['dateTimeMeasured'],
                 df_ug_ref1['chi_p_626'],
                 yerr=df_ug_ref1['chi_p_626_error'],
                 fmt='o',
                 mec=c_light,
                 mew=0.5,
                 ms=5,
                 c=f"{c_light}80",
                 label=lab_ref1_ug)
axes[1].errorbar(df_ug_ref2['dateTimeMeasured'],
                 df_ug_ref2['chi_p_626'],
                 yerr=df_ug_ref2['chi_p_626_error'],
                 fmt='o',
                 mec=c_heavy,
                 mew=0.5,
                 ms=5,
                 c=f"{c_heavy}80",
                 label=lab_ref2_ug)

axes[1].plot(pd.to_datetime(df_ug_fit['common_x'], unit='s'), df_ug_fit['chi_p_626'],
             '-', c="k", lw=2)

axes[1].set_ylabel('$\chi\prime^{smp}_{626, meas}$ (µmol mol$^{-1}$)')


# Subplot C: Cell pressure
axes[2].errorbar(df_ug_ref1['dateTimeMeasured'],
                 df_ug_ref1['cell_pressure'],
                 yerr=df_ug_ref1['cell_pressure_error'],
                 fmt='o',
                 mec=c_light,
                 mew=0.5,
                 ms=5,
                 c=f"{c_light}80",
                 label=lab_ref1_ug)
axes[2].errorbar(df_ug_ref2['dateTimeMeasured'],
                 df_ug_ref2['cell_pressure'],
                 yerr=df_ug_ref2['cell_pressure_error'],
                 fmt='o',
                 mec=c_heavy,
                 mew=0.5,
                 ms=5,
                 c=f"{c_heavy}80",
                 label=lab_ref2_ug)

axes[2].plot(pd.to_datetime(df_ug_fit['common_x'], unit='s'), df_ug_fit['cell_pressure'],
             '-', c="k", lw=2)

axes[2].set_ylabel(r'$\mathit{P}_{cell}$ (Pa)')

# Subplot D: Cell temperature
axes[3].errorbar(df_ug_ref1['dateTimeMeasured'],
                 df_ug_ref1['cell_temperature'],
                 yerr=df_ug_ref1['cell_temperature_error'],
                 fmt='o',
                 mec=c_light,
                 mew=0.5,
                 ms=5,
                 c=f"{c_light}80",
                 label=lab_ref1_ug)
axes[3].errorbar(df_ug_ref2['dateTimeMeasured'],
                 df_ug_ref2['cell_temperature'],
                 yerr=df_ug_ref2['cell_temperature_error'],
                 fmt='o',
                 mec=c_heavy,
                 mew=0.5,
                 ms=5,
                 c=f"{c_heavy}80",
                 label=lab_ref2_ug)

axes[3].plot(pd.to_datetime(df_ug_fit['common_x'], unit='s'), df_ug_fit['cell_temperature'],
             '-', c="k", lw=2)

axes[3].set_ylabel(r'$\mathit{T}_{cell}$ (K)')

# Subplot E: Electronics temperature
axes[4].errorbar(df_ug_ref1['dateTimeMeasured'],
                 df_ug_ref1['electronics_temperature'],
                 yerr=df_ug_ref1['electronics_temperature_error'],
                 fmt='o',
                 mec=c_light,
                 mew=0.5,
                 ms=5,
                 c=f"{c_light}80",
                 label=lab_ref1_ug)
axes[4].errorbar(df_ug_ref2['dateTimeMeasured'],
                 df_ug_ref2['electronics_temperature'],
                 yerr=df_ug_ref2['electronics_temperature_error'],
                 fmt='o',
                 mec=c_heavy,
                 mew=0.5,
                 ms=5,
                 c=f"{c_heavy}80",
                 label=lab_ref2_ug)

axes[4].plot(pd.to_datetime(df_ug_fit['common_x'], unit='s'), df_ug_fit['electronics_temperature'],
             '-', c="k", lw=2)

axes[4].set_ylabel(r'$\mathit{T}_{elec}$ (K)')


# Subplot - Compression in ∆17O
def format_text(text):
    return text.replace("$", "").replace(" ", r"\ ")
lab_ref1 = format_text(lab_ref1_ug)
lab_ref2 = format_text(lab_ref2_ug)

axes[5].plot(pd.to_datetime(df_ug_fit['common_x'], unit='s'),
             df_ug_fit["Dp17O_compression"],
             '-', c="k", lw=2,
             label=rf"$\Delta\prime^{{17}}O_{{{lab_ref1}}} - \Delta\prime^{{17}}O_{{{lab_ref2}}}$")

axes[5].set_ylabel("Compression (ppm)")

# Predictor model
predictors = ["cell_temperature", "electronics_temperature", "cell_pressure", "chi_p_626"]
X = add_constant(df_ug_fit[predictors], has_constant="add")
y = df_ug_fit["Dp17O_compression"]
model = OLS(y, X).fit()
df_ug_fit["compression_predicted"] = model.predict(X)

axes[5].plot(pd.to_datetime(df_ug_fit['common_x'], unit='s'),
             df_ug_fit["compression_predicted"],
             label=r"$\mathcal{f}$($\it{T}$$_{cell}$, $\it{T}$$_{elec}$, $\it{P}$$_{cell}$, $\chi\prime_{626}$)", linestyle="--", linewidth=2, c="#1455C0")

axes[5].legend(loc='lower right')

axes[-1].set_xlabel('Measurement date')
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "LT_Figure_7.png"))
plt.close("all")



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Figure 8 - University of Cape Town

# Apply LOESS smoothing for University of Cape Town data
df_uct_fit, df_uct_ref1, df_uct_ref2, lab_ref1_uct, lab_ref2_uct = loess_on_columns(df_uct, "NBS18", "IAEA603")

fig, axes = plt.subplots(5, 1, sharex=True, figsize=(6, 7.5))
axes = axes.flatten()

for ax in fig.get_axes():
    i = fig.get_axes().index(ax)
    ax.text(0.01, 0.975, chr(97 + i),
            size=12, weight="bold", ha="left", va="top",
            transform=ax.transAxes)

# Subplot -  ∆'17O data
axes[0].errorbar(df_uct_ref1['dateTimeMeasured'],
                 (df_uct_ref1['Dp17O']),
                 yerr=df_uct_ref1['Dp17O_error'],
                 fmt='o',
                 mec=c_NBS18,
                 mew=0.5,
                 ms=5,
                 c=f"{c_NBS18}80",
                 label=lab_ref1_uct)
axes[0].errorbar(df_uct_ref2['dateTimeMeasured'],
                 (df_uct_ref2['Dp17O']),
                 yerr=df_uct_ref2['Dp17O_error'],
                 fmt='o',
                 mec=c_IAEA603,
                 mew=0.5,
                 ms=5,
                 c=f"{c_IAEA603}80",
                 label=lab_ref2_uct)

axes[0].plot(pd.to_datetime(df_uct_fit['common_x'], unit='s'), (df_uct_fit['Dp17O_ref1']),
             '-', c=c_NBS18, lw=2)
axes[0].plot(pd.to_datetime(df_uct_fit['common_x'], unit='s'), (df_uct_fit['Dp17O_ref2']),
             '-', c=c_IAEA603, lw=2)

axes[0].plot([], [], '-', c="k", lw=2, label="LOESS fit")
axes[0].legend(loc="upper right")

axes[0].set_ylabel("$\Delta\prime^{17}O^{smp/wg}_{true}$ (ppm)")


# Subplot C: Cell pressure
axes[1].errorbar(df_uct_ref1['dateTimeMeasured'],
                 df_uct_ref1['cell_pressure'],
                 yerr=df_uct_ref1['cell_pressure_error'],
                 fmt='o',
                 mec=c_NBS18,
                 mew=0.5,
                 ms=5,
                 c=f"{c_NBS18}80",
                 label=lab_ref1_uct)
axes[1].errorbar(df_uct_ref2['dateTimeMeasured'],
                 df_uct_ref2['cell_pressure'],
                 yerr=df_uct_ref2['cell_pressure_error'],
                 fmt='o',
                 mec=c_IAEA603,
                 mew=0.5,
                 ms=5,
                 c=f"{c_IAEA603}80",
                 label=lab_ref2_uct)

axes[1].plot(pd.to_datetime(df_uct_fit['common_x'], unit='s'), df_uct_fit['cell_pressure'],
             '-', c="k", lw=2)

axes[1].set_ylabel(r'$\mathit{P}_{cell}$ (Pa)')

# Subplot D: Cell temperature
axes[2].errorbar(df_uct_ref1['dateTimeMeasured'],
                 df_uct_ref1['cell_temperature'],
                 yerr=df_uct_ref1['cell_temperature_error'],
                 fmt='o',
                 mec=c_NBS18,
                 mew=0.5,
                 ms=5,
                 c=f"{c_NBS18}80",
                 label=lab_ref1_ug)
axes[2].errorbar(df_uct_ref2['dateTimeMeasured'],
                 df_uct_ref2['cell_temperature'],
                 yerr=df_uct_ref2['cell_temperature_error'],
                 fmt='o',
                 mec=c_IAEA603,
                 mew=0.5,
                 ms=5,
                 c=f"{c_IAEA603}80",
                 label=lab_ref2_ug)

axes[2].plot(pd.to_datetime(df_uct_fit['common_x'], unit='s'), df_uct_fit['cell_temperature'],
             '-', c="k", lw=2)

axes[2].set_ylabel(r'$\mathit{T}_{cell}$ (K)')

# Subplot E: Electronics temperature
axes[3].errorbar(df_uct_ref1['dateTimeMeasured'],
                 df_uct_ref1['electronics_temperature'],
                 yerr=df_uct_ref1['electronics_temperature_error'],
                 fmt='o',
                 mec=c_NBS18,
                 mew=0.5,
                 ms=5,
                 c=f"{c_NBS18}80",
                 label=lab_ref1_ug)
axes[3].errorbar(df_uct_ref2['dateTimeMeasured'],
                 df_uct_ref2['electronics_temperature'],
                 yerr=df_uct_ref2['electronics_temperature_error'],
                 fmt='o',
                 mec=c_IAEA603,
                 mew=0.5,
                 ms=5,
                 c=f"{c_IAEA603}80",
                 label=lab_ref2_ug)

axes[3].plot(pd.to_datetime(df_uct_fit['common_x'], unit='s'), df_uct_fit['electronics_temperature'],
             '-', c="k", lw=2)

axes[3].set_ylabel(r'$\mathit{T}_{elec}$ (K)')


# Subplot F: Scale compression in ∆17O
def format_text(text):
    return text.replace("$", "").replace(" ", r"\ ")
lab_ref1 = format_text(lab_ref1_uct)
lab_ref2 = format_text(lab_ref2_uct)

axes[4].plot(pd.to_datetime(df_uct_fit['common_x'], unit='s'),
             df_uct_fit["Dp17O_compression"],
             '-', c="k", lw=2,
             label=rf"$\Delta\prime^{{17}}O_{{{lab_ref1}}} - \Delta\prime^{{17}}O_{{{lab_ref2}}}$")

axes[4].set_ylabel("Compression (ppm)")

# Predictor model
predictors = ["cell_temperature", "electronics_temperature", "cell_pressure"]
X = add_constant(df_uct_fit[predictors], has_constant="add")
y = df_uct_fit["Dp17O_compression"]
model = OLS(y, X).fit()
df_uct_fit["compression_predicted"] = model.predict(X)

axes[4].plot(pd.to_datetime(df_uct_fit['common_x'], unit='s'),
             df_uct_fit["compression_predicted"],
             label=r"$\mathcal{f}$($\it{T}$$_{cell}$, $\it{T}$$_{elec}$, $\it{P}$$_{cell}$)",
             ls="--", lw=2, c="#814997")

axes[4].legend(loc='upper right')

axes[-1].set_xlabel('Measurement date')
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "LT_Figure_8.png"))
plt.close("all")



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Figure 9 - R squared of predictor combinations

label_map = {
    "cell_temperature": r"$\it{T}$$_{cell}$",
    "electronics_temperature": r"$\it{T}$$_{elec}$",
    "cell_pressure": r"$\it{P}$$_{cell}$",
    "chi_p_626": r"$\chi\prime_{626}$"
}

def format_predictor_label(combo):
    return " + ".join([label_map[p] for p in combo])

def run_mlr(df, predictors):
    """Run multiple linear regressions for all predictor combinations 
    against the constant target Dp17O_compression."""
    results = []
    y = df["Dp17O_compression"]
    for k in range(1, len(predictors) + 1):
        for combo in combinations(predictors, k):
            X = add_constant(df[list(combo)], has_constant="add")
            model = OLS(y, X).fit()
            results.append((
                format_predictor_label(combo),
                model.rsquared,
                model.f_pvalue
            ))
    return results

# Göttingen dataset
ug_mlr_results = run_mlr(
    df_ug_fit,
    ["cell_temperature", "electronics_temperature", "cell_pressure", "chi_p_626"]
)

# Cape Town dataset
uct_mlr_results = run_mlr(
    df_uct_fit,
    ["cell_temperature", "electronics_temperature", "cell_pressure"]
)

# Convert to DataFrames for plotting
ug_df = pd.DataFrame(ug_mlr_results, columns=["Predictors", "R2", "pval"]).sort_values("R2", ascending=False)
uct_df = pd.DataFrame(uct_mlr_results, columns=["Predictors", "R2", "pval"])

# Reindex Cape Town results to match Göttingen combinations
uct_df = ug_df[["Predictors"]].merge(uct_df, on="Predictors", how="left")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.5), sharex = True, sharey = True)
axes = axes.flatten()

for ax in fig.get_axes():
    i = fig.get_axes().index(ax)
    ax.text(0.98, 0.975, chr(97 + i),
            size=12, weight="bold", ha="right", va="top",
            transform=ax.transAxes)

bars = axes[0].barh(ug_df["Predictors"], ug_df["R2"], color="#1455C0")
for bar, r2, pval in zip(bars, ug_df["R2"], ug_df["pval"]):
    label = f"$R^2$={r2:.2g}"
    value = bar.get_width()
    if value < 0.4:
        # Place label outside the bar, to the right
        axes[0].text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            label,
            va='center',
            ha='left',
            fontsize=6
        )
    else:
        # Place label inside the bar, near the left edge
        axes[0].text(
            bar.get_width() - 0.01,
            bar.get_y() + bar.get_height() / 2,
            label,
            va='center',
            ha='right',
            fontsize=6,
            color='white'
        )
    if pval >= 0.05:
        print(f"Warning: Predictor combination '{ug_df['Predictors'].iloc[i]}' has p-value {pval:.1g}")



axes[0].set_title("University of Göttingen")
axes[0].set_xlabel(r"$R^{2}$")
axes[0].set_ylabel("Predictor combination")
axes[0].set_xlim(0, 1)
axes[0].tick_params(axis="y", length=0) 

bars = axes[1].barh(uct_df["Predictors"], uct_df["R2"], color="#814997")
for i, (bar, r2, pval) in enumerate(zip(bars, uct_df["R2"], uct_df["pval"])):
    y = bar.get_y() + bar.get_height() / 2
    if pd.isna(r2):
        axes[1].text(0.01, y, r"$\it{not\ considered}$", va='center', ha='left', fontsize=6, color='gray')
    else:
        label = f"$R^2$={r2:.2g}"
        axes[1].text(bar.get_width() + 0.01, y, label, va='center', ha='left', fontsize=6)
        if pval >= 0.05:
            print(
                f"Warning: Predictor combination '{uct_df['Predictors'].iloc[i]}' has p-value {pval:.1g}")

axes[1].set_title("University of Cape Town")
axes[1].set_xlabel(r"$R^{2}$")
axes[1].tick_params(axis="y", length=0) 
        
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "LT_Figure_9.png"))
plt.close("all")