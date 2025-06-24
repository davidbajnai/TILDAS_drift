'''
This script retrieves absorption coefficients from the HITRAN database, and visualizes the results in a plot.
It also imports experimental data from a CSV file and plots it alongside the HITRAN data.

INPUT:
- 250521_003216_001_SIG.csv: A CSV file containing spectral data retreived from a correspondingly named .spb file.
- HITRAN database files (optional): Required for calculating absorption coefficients.
        If not present, the script will attempt to fetch them.

OUTPUT:
- LT_Figure_1.png: A plot showing the absorption coefficients from HITRAN and the experimental data.
'''

# Import libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from hapi import *

# Get script directory for relative file paths
script_dir = os.path.dirname(os.path.abspath(__file__))

# Get data directory
data_dir = os.path.join(script_dir, '../data')

# Get figures directory
figures_dir = os.path.join(script_dir, '../figures')
 
# Plot parameters
plt.rcParams.update({"font.size": 8})
plt.rcParams["figure.figsize"] = (3.5, 5)
plt.rcParams["patch.linewidth"] = 1
plt.rcParams['lines.linewidth'] = 1
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams['savefig.transparent'] = False
plt.rcParams['mathtext.default'] = 'regular'
plt.rcParams['legend.loc'] = 'best'

# Import experimental data
data = pd.read_csv(os.path.join(data_dir, "250521_003216_001_SIG.csv"))

# Get data from HITRAN database
db_begin(os.path.join(data_dir, 'HITRAN'))

# Check if CO2 table is present
if 'CO2' not in tableList():
    fetch_by_ids('CO2', [7, 8, 9, 10], 2348, 2351)
if "H2O" not in tableList():
    fetch_by_ids('H2O', [1, 2], 2348, 2351)
if "N2O" not in tableList():
    fetch_by_ids('N2O', [1, 2, 3, 4, 5], 2348, 2351)
if "NO2" not in tableList():
    fetch_by_ids('NO2', [1, 2, 3], 2348, 2351)
if "CO" not in tableList():
    fetch_by_ids('CO', [1, 2, 3], 2348, 2351)
if "SO2" not in tableList():
    fetch_by_ids('SO2', [1, 2, 3], 2348, 2351)

def get_co2_voigt_abscoef(iso_N, xmin=2348, xmax=2351,
                          cell_pressure=41.3335, cell_temperature=297.632,
                          diluent_self_ppm=420, db_table='CO2'):
    pressure = cell_pressure / 760  # Torr to atm
    temperature = cell_temperature
    nu, coef = absorptionCoefficient_Voigt(
        SourceTables=db_table,
        Components=[(2, iso_N)],
        Environment={'T': temperature, 'p': pressure},
        WavenumberRange=[xmin, xmax],
        WavenumberStep=(xmax - xmin) / 10000,
        Diluent={'self': 1 * diluent_self_ppm / 1e6, 'air': 1 - diluent_self_ppm / 1e6},
        HITRAN_units=False
    )
    return nu, coef

def get_h2o_voigt_abscoef(iso_N, xmin=2348, xmax=2351,
                          cell_pressure=41.3335, cell_temperature=297.632,
                          diluent_self_ppm=1000, db_table='H2O'):
    pressure = cell_pressure / 760  # Torr to atm
    temperature = cell_temperature
    nu, coef = absorptionCoefficient_Voigt(
        SourceTables=db_table,
        Components=[(1, iso_N)],
        Environment={'T': temperature, 'p': pressure},
        WavenumberRange=[xmin, xmax],
        WavenumberStep=(xmax - xmin) / 10000,
        Diluent={'self': 1 * diluent_self_ppm / 1e6, 'air': 1 - diluent_self_ppm / 1e6},
        HITRAN_units=False
    )
    return nu, coef


def get_n2o_voigt_abscoef(iso_N, xmin=2348, xmax=2351,
                          cell_pressure=41.3335, cell_temperature=297.632,
                          diluent_self_ppm=1000, db_table='N2O'):
    pressure = cell_pressure / 760  # Torr to atm
    temperature = cell_temperature
    nu, coef = absorptionCoefficient_Voigt(
        SourceTables=db_table,
        Components=[(4, iso_N)],
        Environment={'T': temperature, 'p': pressure},
        WavenumberRange=[xmin, xmax],
        WavenumberStep=(xmax - xmin) / 10000,
        Diluent={'self': 1 * diluent_self_ppm / 1e6, 'air': 1 - diluent_self_ppm / 1e6},
        HITRAN_units=False
    )
    return nu, coef

# Get Voigt absorption coefficients for CO isotopologues
def get_co_voigt_abscoef(iso_N, xmin=2348, xmax=2351,
                         cell_pressure=41.3335, cell_temperature=297.632,
                         diluent_self_ppm=1000, db_table='CO'):
    pressure = cell_pressure / 760  # Torr to atm
    temperature = cell_temperature
    nu, coef = absorptionCoefficient_Voigt(
        SourceTables=db_table,
        Components=[(1, iso_N)],
        Environment={'T': temperature, 'p': pressure},
        WavenumberRange=[xmin, xmax],
        WavenumberStep=(xmax - xmin) / 10000,
        Diluent={'self': 1 * diluent_self_ppm / 1e6, 'air': 1 - diluent_self_ppm / 1e6},
        HITRAN_units=False
    )
    return nu, coef

# Get Voigt absorption coefficients for SO2 isotopologues
def get_so2_voigt_abscoef(iso_N, xmin=2348, xmax=2351,
                          cell_pressure=41.3335, cell_temperature=297.632,
                          diluent_self_ppm=1000, db_table='SO2'):
    pressure = cell_pressure / 760  # Torr to atm
    temperature = cell_temperature
    nu, coef = absorptionCoefficient_Voigt(
        SourceTables=db_table,
        Components=[(5, iso_N)],
        Environment={'T': temperature, 'p': pressure},
        WavenumberRange=[xmin, xmax],
        WavenumberStep=(xmax - xmin) / 10000,
        Diluent={'self': 1 * diluent_self_ppm / 1e6, 'air': 1 - diluent_self_ppm / 1e6},
        HITRAN_units=False
    )
    return nu, coef

# Get Voigt absorption coefficients for CO2 isotopologues
x626, y626 = get_co2_voigt_abscoef(1)   # 626
x627, y627 = get_co2_voigt_abscoef(4)   # 627
x628, y628 = get_co2_voigt_abscoef(3)   # 628

# Get Voigt absorption coefficients for H2O isotopologues
x1, y1 = get_h2o_voigt_abscoef(1)   # 161
x2, y2 = get_h2o_voigt_abscoef(2)   # 181

# Get Voigt absorption coefficients for N2O isotopologues
x4, y4 = get_n2o_voigt_abscoef(1)   # 
x5, y5 = get_n2o_voigt_abscoef(2)   # 2
x6, y6 = get_n2o_voigt_abscoef(3)   # 3
# Get Voigt absorption coefficients for CO isotopologues
x7, y7 = get_co_voigt_abscoef(1)   # 1
x8, y8 = get_co_voigt_abscoef(2)   # 2
x9, y9 = get_co_voigt_abscoef(3)   # 3
# Get Voigt absorption coefficients for SO2 isotopologues
x10, y10 = get_so2_voigt_abscoef(1)   # 1
x11, y11 = get_so2_voigt_abscoef(2)   # 2
x12, y12 = get_so2_voigt_abscoef(3)   # 3


# Make the plot
fig, axes = plt.subplots(ncols = 1, nrows=2, sharex=True)
axes = axes.flatten()

for ax in fig.get_axes():
    i = fig.get_axes().index(ax)
    ax.text(0.02, 0.98, chr(97 + i),
            size=12, weight="bold", ha="left", va="top",
            bbox=dict(fc="w", pad=0.1, ec="none", alpha=0.8),
            transform=ax.transAxes)

# Only unique legend entries:
handles, labels = axes[0].get_legend_handles_labels()
from collections import OrderedDict
by_label = OrderedDict(zip(labels, handles))
axes[0].legend(by_label.values(), by_label.keys())


axes[0].plot(x626, y626, color='k', lw=1, label='"626" ($^{16}O^{12}C^{16}O$)')
axes[0].plot(x628, y628, color='r', lw=1, label='"628" ($^{16}O^{12}C^{18}O$)')
axes[0].plot(x627, y627, color='g', lw=1, label='"627" ($^{16}O^{12}C^{17}O$)')

# axes[0].plot(x1, y1, color='b', lw=1, label='"H2O 1" ($^{16}O^{16}O^{16}O$)')
# axes[0].plot(x2, y2, color='m', lw=1, label='"H2O 2" ($^{16}O^{16}O^{18}O$)')
# axes[0].plot(x4, y4, color='c', lw=1, label='"N2O 1" ($^{14}N^{14}N^{16}O$)')
# axes[0].plot(x5, y5, color='y', lw=1, label='"N2O 2" ($^{14}N^{14}N^{18}O$)')
# axes[0].plot(x6, y6, color='orange', lw=1, label='"N2O 3" ($^{14}N^{15}N^{16}O$)')
# axes[0].plot(x7, y7, color='purple', lw=1, label='"CO 1" ($^{12}C^{16}O$)')
# axes[0].plot(x8, y8, color='brown', lw=1, label='"CO 2" ($^{12}C^{17}O$)')
# axes[0].plot(x9, y9, color='pink', lw=1, label='"CO 3" ($^{12}C^{18}O$)')
# axes[0].plot(x10, y10, color='gray', lw=1, label='"SO2 1" ($^{32}S^{16}O^{16}O$)')
# axes[0].plot(x11, y11, color='olive', lw=1, label='"SO2 2" ($^{32}S^{16}O^{18}O$)')
# axes[0].plot(x12, y12, color='teal', lw=1, label='"SO2 3" ($^{32}S^{18}O^{16}O$)')

axes[0].legend()
axes[0].set_ylim(0, 0.6)
axes[0].set_xlim(2349.15, 2349.45)

axes[0].set_ylabel("Absorbtion coefficient (cm$^{-2}$)")
axes[0].xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
axes[0].xaxis.set_major_locator(ticker.MultipleLocator(0.1))


axes[1].scatter(data['wb1_frequency'], data['wb1_spectrum'],
                marker="o", s=10, c="#63A615", label='measurement')
axes[1].plot(data['wb1_frequency'], data['wb1_wintel_fit'],
             c="#1455C0", label='spectral fit')
axes[1].plot(data['wb1_frequency'], data['wb1_wintel_base'],
             ls="-", c="#F39200", label='baseline')

axes[1].legend()
axes[1].set_ylim(-260, 10)
axes[1].set_xlabel('Wavenumber (cm$^{-1}$)')
axes[1].set_ylabel('Signal (mV)')

plt.tight_layout()
plt.savefig(os.path.join(figures_dir, "LT_Figure_1.png"))
plt.close("all")