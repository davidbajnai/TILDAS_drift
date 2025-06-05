import numpy as np
import pandas as pd
import statsmodels.api as sm

def prime(x):
    return 1000 * np.log(x / 1000 + 1)


def unprime(x):
    return (np.exp(x / 1000) - 1) * 1000


def Dp17O(d17O, d18O):
    return (prime(d17O) - 0.528 * prime(d18O)) * 1000


def d17O(d18O, Dp17O):
    return unprime(Dp17O / 1000 + 0.528 * prime(d18O))


def mix_d17O(d18O_A, d17O_A=None, D17O_A=None, d18O_B=None, d17O_B=None, D17O_B=None, step=100):
    ratio_B = np.arange(0, 1+1/step, 1/step)

    if d17O_A is None:
        d17O_A = unprime(D17O_A/1000 + 0.528 * prime(d18O_A))

    if d17O_B is None:
        d17O_B = unprime(D17O_B/1000 + 0.528 * prime(d18O_B))

    mix_d18O = ratio_B * float(d18O_B) + (1 - ratio_B) * float(d18O_A)
    mix_d17O = ratio_B * float(d17O_B) + (1 - ratio_B) * float(d17O_A)
    mix_D17O = Dp17O(mix_d17O, mix_d18O)
    xB = ratio_B * 100

    df = pd.DataFrame(
        {'mix_d17O': mix_d17O, 'mix_d18O': mix_d18O, 'mix_Dp17O': mix_D17O, 'xB': xB})
    return df


def scale_series(series, min_value, max_value):
    series_min = series.min()
    series_max = series.max()
    
    scaled_series = ((series - series_min) / (series_max - series_min)) * (max_value - min_value) + min_value
    return scaled_series


def apply_loess(df, column_to_fit, common_x, filter_for_sample_name=None):
    """
    Applies LOESS smoothing to a specified column in a DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the data.
        column_to_fit (str): The name of the column to apply LOESS smoothing on.
        common_x (array-like): The common x-values for interpolation.
        filter_for_sample_name (str, optional): Filter condition for 'SampleName' column.

    Returns:
        dict: A dictionary containing the smoothed values.
    """

    # Apply filter if provided
    if filter_for_sample_name:
        df = df[df['SampleName'].str.contains(filter_for_sample_name, regex=False)]

    # Check if the column exists in the DataFrame
    if column_to_fit not in df.columns:
        raise ValueError(f"Column '{column_to_fit}' not found in the DataFrame.")

    # Apply LOESS smoothing on the specified column
    loess_fit = sm.nonparametric.lowess(
        df[column_to_fit], df["dateTimeMeasured_num"], frac=0.3
    )

    # Interpolation for smoother plotting and matching data lengths
    loess_fit_interp = np.interp(common_x, loess_fit[:, 0], loess_fit[:, 1])

    # Return the results
    return {"loess_fit": loess_fit_interp}


def torr_to_pascal(torr):
    return torr * 133.322


def pascal_to_torr(pascal):
    return pascal / 133.322


def celsius_to_kelvin(celsius):
    return celsius + 273.15


def kelvin_to_celcius(kelvin):
    return kelvin - 273.15


def mole_fraction_to_concentration(chi_frac, T_K, P_Pa, iso):
    """
    Convert mole fractions (chi) to concentration (C)
    All units are in SI.
    """
    R = 8.31446261815324

    if iso == 626:
        X = 0.98420
    elif iso == 628:
        X = 0.0039471
    elif iso == 627:
        X = 0.000734 

    # Ensure NaNs propagate row-wise
    result = (chi_frac * X * P_Pa) / (R * T_K)
    return result


def concentration_to_mole_fraction(C, T_K, P_Pa, iso):
    """
    Convert concentration (C) to mole fractions (chi)
    All units are in SI
    """
    R = 8.31446261815324

    if iso == 626:
        X = 0.98420
    elif iso == 628:
        X = 0.0039471
    elif iso == 627:
        X = 0.000734 
    return (C * R * T_K) / (X * P_Pa)

def f_drift(chi_626=None, T=None, P=None, A=1.0, B=1.0, C=1.0):
    """
    Generalized equation for parameterizing the drift.
    Equation 8 of the manuscript.
    """

    term1 = A * np.log(chi_626/10**6)
    term2 = B * np.log(T)
    term3 = C * np.log(P)

    return term1 + term2 + term3


def deltaO_vs_WG(deltaO_sample, deltaO_wg):
    return (deltaO_sample - deltaO_wg)/(1000+deltaO_wg) * 1000


def devmean(x):
    return x - x.mean()
