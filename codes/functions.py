import numpy as np

def prime(x):
    return 1000 * np.log(x / 1000 + 1)


def unprime(x):
    return (np.exp(x / 1000) - 1) * 1000


def Dp17O(d17O, d18O):
    return (prime(d17O) - 0.528 * prime(d18O)) * 1000


def d17O(d18O, Dp17O):
    return unprime(Dp17O / 1000 + 0.528 * prime(d18O))


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


def deltaO_vs_WG(deltaO_sample, deltaO_wg):
    return (deltaO_sample - deltaO_wg)/(1000+deltaO_wg) * 1000


def devmean(x):
    return x - x.mean()