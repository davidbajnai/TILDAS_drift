"""
This sccript shows the long-term drift of the of ∆17O measurements.

INPUT:
- LT Table S1.csv: A CSV file containing data from the University of Göttingen.
- LT Table S2.csv: A CSV file containing data from the University of Cape Town.

OUTPUT:
- LT Figure 7.png: A plot showing the long-term drift of ∆17O measurements at the University of Göttingen.
- LT Figure 8.png: A plot showing the correlation coefficients from the multivariate linear regression models.
- LT Figure 9.png: A plot showing the long-term drift of ∆17O measurements at the University of Cape Town.
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
from sklearn.linear_model import LinearRegression
from cmap import Colormap
from matplotlib.colors import Normalize
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
plt.rcParams["savefig.dpi"] = 800
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams['savefig.transparent'] = False
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['legend.loc'] = 'upper right'

# Colors for plotting the reference materials
c_NBS18, c_IAEA603 = "#F75056", "#641E32"
c_light, c_heavy = "#309FD1", "#0C3992"

# Get data - University of Cape Town
df_uct = pd.read_csv(os.path.join(data_dir, "LT Table S2.csv"))
df_uct['dateTimeMeasured'] = pd.to_datetime(df_uct['Name'], format='%y%m%d_%H%M%S', errors='coerce')
df_uct['dateTimeMeasured_num'] = df_uct["dateTimeMeasured"].astype(np.int64) // 10**9
df_uct = df_uct.sort_values(by=["dateTimeMeasured"], ascending=True, ignore_index=True)

# Make column names consistent for plotting
df_uct["SampleName"] = df_uct['AnalyticalID']
df_uct['Dp17O'] = df_uct['Dp17O_corr']*1000
df_uct['Dp17O_error'] = df_uct['Dp17O_raw_std']*1000
df_uct['cell_temperature'] = df_uct['Traw_mean']
df_uct['cell_temperature_error'] = df_uct['Traw_std']
df_uct['cell_pressure'] = torr_to_pascal(df_uct['Praw_mean'])
df_uct['cell_pressure_error'] = df_uct['Praw_std']*133.322
df_uct['chi_p_626'] = 421
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
df_ug = pd.read_csv(os.path.join(data_dir, "LT Table S1.csv"))
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
df_ug['chi_p_626'] = df_ug['pCO2Ref'] # in ppmv
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
    df_fit["chi_p_626"] = apply_loess(df, "chi_p_626", common_x)["loess_fit"]

    df_fit["Dp17O_compression"] = (df_fit["Dp17O_ref1"] - df_fit["Dp17O_ref2"])

    df_ug_ref1 = df[df['SampleName'].str.contains(lab_ref1, regex=False)]
    df_ref2 = df[df['SampleName'].str.contains(lab_ref2, regex=False)]

    return df_fit, df_ug_ref1, df_ref2, lab_ref1, lab_ref2



# Figure 8 - University of Göttingen

# Apply LOESS smoothing for University of Göttingen data
df_ug_fit, df_ug_ref1, df_ug_ref2, lab_ref1_ug, lab_ref2_ug = loess_on_columns(df_ug, "light $CO_2$", "heavy $CO_2$")

fig, axes = plt.subplots(5, 1, sharex=True, figsize=(6, 8))
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

axes[0].set_ylabel("$\Delta\prime^{17}O_{smp/std}$ (ppm)")

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

axes[1].set_ylabel('$\chi\prime^{sam}_{626}$ (µmol mol$^{-1}$)')


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
axes[2].set_ylabel(r'$\mathit{P}$ (Pa)')

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

axes[3].set_ylabel(r'$\mathit{T}$ (K)')


# Subplot - Compression in ∆17O
def format_text(text):
    return text.replace("$", "").replace(" ", r"\ ")
lab_ref1 = format_text(lab_ref1_ug)
lab_ref2 = format_text(lab_ref2_ug)

axes[4].plot(pd.to_datetime(df_ug_fit['common_x'], unit='s'),
             df_ug_fit["Dp17O_compression"],
             '-', c="k", lw=2,
             label=rf"$\Delta\prime^{{17}}O_{{{lab_ref1}}} - \Delta\prime^{{17}}O_{{{lab_ref2}}}$")

axes[4].set_ylabel("Compression (ppm)")


X = df_ug_fit[["cell_temperature", "cell_pressure", "chi_p_626"]]
y = df_ug_fit["Dp17O_compression"]

for k in range(1, 4):
    for combo in combinations(["cell_temperature", "cell_pressure", "chi_p_626"], k):
        X = df_ug_fit[list(combo)]
        model = LinearRegression().fit(X, y)
        r2 = model.score(X, y)

# Define predictors and target for the plot
predictors = ["cell_temperature", "cell_pressure"]
X = df_ug_fit[predictors]
y = df_ug_fit["Dp17O_compression"]

# Fit model
model = LinearRegression().fit(X, y)
df_ug_fit["compression_predicted"] = model.predict(X)

axes[4].plot(pd.to_datetime(df_ug_fit['common_x'], unit='s'),
             df_ug_fit["compression_predicted"],
             label=r"$\it{f}$($\it{T}$,$\it{P}$)", linestyle="--", linewidth=2, c="#1455C0")

axes[4].legend(loc='lower right')

axes[-1].set_xlabel('Measurement date')
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "LT Figure 7.png"))
plt.close("all")




# Figure 9 - University of Cape Town

# Apply LOESS smoothing for University of Cape Town data
df_uct_fit, df_uct_ref1, df_uct_ref2, lab_ref1_uct, lab_ref2_uct = loess_on_columns(df_uct, "NBS18", "IAEA603")

fig, axes = plt.subplots(4, 1, sharex=True, figsize=(6, 7))
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

axes[0].set_ylabel("$\Delta\prime^{17}O_{smp/std}$ (ppm)")


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
axes[1].set_ylabel(r'$\mathit{P}$ (Pa)')

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

axes[2].set_ylabel(r'$\mathit{T}$ (K)')


# Subplot - Compression in ∆17O
def format_text(text):
    return text.replace("$", "").replace(" ", r"\ ")
lab_ref1 = format_text(lab_ref1_uct)
lab_ref2 = format_text(lab_ref2_uct)

axes[3].plot(pd.to_datetime(df_uct_fit['common_x'], unit='s'),
             df_uct_fit["Dp17O_compression"],
             '-', c="k", lw=2,
             label=rf"$\Delta\prime^{{17}}O_{{{lab_ref1}}} - \Delta\prime^{{17}}O_{{{lab_ref2}}}$")

axes[3].set_ylabel("Compression (ppm)")


X = df_uct_fit[["cell_temperature", "cell_pressure", "chi_p_626"]]
y = df_uct_fit["Dp17O_compression"]

for k in range(1, 4):
    for combo in combinations(["cell_temperature", "cell_pressure", "chi_p_626"], k):
        X = df_uct_fit[list(combo)]
        model = LinearRegression().fit(X, y)
        r2 = model.score(X, y)

# Define predictors and target for the plot
predictors = ["cell_temperature", "cell_pressure"]
X = df_uct_fit[predictors]
y = df_uct_fit["Dp17O_compression"]

# Fit model
model = LinearRegression().fit(X, y)
df_uct_fit["compression_predicted"] = model.predict(X)

axes[3].plot(pd.to_datetime(df_uct_fit['common_x'], unit='s'),
             df_uct_fit["compression_predicted"],
             label=r"$\it{f}$($\it{T}$,$\it{P}$)", linestyle="--", linewidth=2, c = "#EC0016")

axes[3].legend(loc='upper right')

axes[-1].set_xlabel('Measurement date')
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "LT Figure 9.png"))
plt.close("all")


# R² of predictor combinations

label_map = {
    "cell_temperature": r"$\it{T}$",
    "cell_pressure": r"$\it{P}$",
    "chi_p_626": r"$\chi\prime_{626}$"
}

def format_predictor_label(combo):
    return " + ".join([label_map[p] for p in combo])

def get_p_value(X, y):
    X_ = sm.add_constant(X)
    model = sm.OLS(y, X_).fit()
    return model.f_pvalue

# Prepare lists of tuples for results
ug_mlr_results = []
uct_mlr_results = []

# Göttingen dataset
for k in range(1, 4):
    for combo in combinations(["cell_temperature", "cell_pressure", "chi_p_626"], k):
        X = df_ug_fit[list(combo)]
        y = df_ug_fit["Dp17O_compression"]
        model = LinearRegression().fit(X, y)
        r2 = model.score(X, y)
        pval = get_p_value(X, y)
        label = format_predictor_label(combo)
        ug_mlr_results.append((label, r2, pval))

# Cape Town dataset
for k in range(1, 3):
    for combo in combinations(["cell_temperature", "cell_pressure"], k):
        X = df_uct_fit[list(combo)]
        y = df_uct_fit["Dp17O_compression"]
        model = LinearRegression().fit(X, y)
        r2 = model.score(X, y)
        pval = get_p_value(X, y)
        label = format_predictor_label(combo)
        uct_mlr_results.append((label, r2, pval))

# Convert to DataFrames for plotting
ug_df = pd.DataFrame(ug_mlr_results, columns=["Predictors", "R²", "pval"]).sort_values("R²", ascending=False)
uct_df = pd.DataFrame(uct_mlr_results, columns=["Predictors", "R²", "pval"])

# Reindex Cape Town results to match Göttingen combinations
uct_df = ug_df[["Predictors"]].merge(uct_df, on="Predictors", how="left")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(6.5, 3), sharex = True, sharey = True)
axes = axes.flatten()

for ax in fig.get_axes():
    i = fig.get_axes().index(ax)
    ax.text(0.98, 0.975, chr(97 + i),
            size=12, weight="bold", ha="right", va="top",
            transform=ax.transAxes)

cmap = Colormap("araucaria_r").to_mpl()  # converts to a Matplotlib-compatible colormap

norm = Normalize(vmin=0, vmax=1)
colors_ug = cmap(norm(ug_df["R²"].values))
bars = axes[0].barh(ug_df["Predictors"], ug_df["R²"], color=colors_ug)
for bar, r2, pval in zip(bars, ug_df["R²"], ug_df["pval"]):
    label = f"$R^2$={r2:.1g}, $\it{{p}}${'<0.05' if pval <= 0.05 else f'={pval:.2g}'}"
    axes[0].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                 label, va='center', ha='left', fontsize=6)

axes[0].set_title("University of Göttingen")
axes[0].set_xlabel("R²")
axes[0].set_ylabel("Predictor combination")
axes[0].set_xlim(0, 1)
axes[0].tick_params(axis="y", length=0) 

colors_uct = cmap(norm(uct_df["R²"].values))
bars = axes[1].barh(uct_df["Predictors"], uct_df["R²"], color=colors_uct)
for i, (bar, r2, pval) in enumerate(zip(bars, uct_df["R²"], uct_df["pval"])):
    y = bar.get_y() + bar.get_height() / 2
    if pd.isna(r2):
        axes[1].text(0.01, y, r"$\it{not\ considered}$", va='center', ha='left', fontsize=6, color='gray')
    else:
        label = f"$R^2$={r2:.1g}, $\it{{p}}${'<0.05' if pval <= 0.05 else f'={pval:.2g}'}"
        axes[1].text(bar.get_width() + 0.01, y, label, va='center', ha='left', fontsize=6)
axes[1].set_title("University of Cape Town")
axes[1].set_xlabel("R²")
axes[1].tick_params(axis="y", length=0) 
        
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "LT Figure 8.png"))
plt.close("all")