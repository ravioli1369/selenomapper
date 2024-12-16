try:
    import tomllib
except ImportError:
    import toml as tomllib
import numpy as np
from scipy.optimize import curve_fit
import numpy.typing as npt
from specutils.spectra.spectrum1d import Spectrum1D
from specutils.fitting import fit_lines
from astropy.modeling import models
from astropy.nddata import StdDevUncertainty
import astropy.units as u
import warnings

warnings.filterwarnings("ignore")


class color:
    """
    Class to define colors for text formatting
    """

    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def display_text(text: str, c: str = "yellow") -> None:
    """
    Display text in a colorful formatted manner

    Args:
        text: The text to be displayed

    Returns:
        None
    """
    colors = {
        "purple": color.PURPLE,
        "cyan": color.CYAN,
        "darkcyan": color.DARKCYAN,
        "blue": color.BLUE,
        "green": color.GREEN,
        "yellow": color.YELLOW,
        "red": color.RED,
    }
    print("#" + "-" * (10 + len(text)) + "#")
    print(
        "#"
        + ("-" * 5)
        + color.BOLD
        + colors[c]
        + color.UNDERLINE
        + str(text)
        + color.END
        + ("-" * 5)
        + "#"
    )
    print("#" + "-" * (10 + len(text)) + "#")


def parse_toml_params(filename: str) -> dict:
    """
    Parse the toml file containing the parameters

    Args:
    filename: The path to the toml file

    Returns:
    The parameters in a dictionary format
    """
    with open(filename) as file:
        return tomllib.loads(file.read())


def _single_gaussian(x, amplitude, mean, sigma):
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma**2))


def _double_gaussian(x, a1, m1, s1, a2, m2, s2):
    return _single_gaussian(x, a1, m1, s1) + _single_gaussian(x, a2, m2, s2)


def detect_lines(
    counts: npt.NDArray,
    err: npt.NDArray,
    energies: npt.NDArray,
    elements: list[str],
    element_energies: dict[str, float],
    fit_func: str,
) -> dict | None:
    """
    Detect the lines in the given spectrum.

    Args:
        counts: Array of counts
        energies: Array of energies
        elements: Elements to look for

    Returns:
        A dictionary containing the parameters of the detected lines and the
        fitted gaussian for each line.
    """
    if len(elements) == 0:
        return None

    if fit_func == "curve_fit":
        return fit_curvefit(energies, counts, err, elements, element_energies)
    else:
        return fit_specutils(energies, counts, err, elements, element_energies)


def fit_curvefit(
    energies: npt.NDArray,
    counts: npt.NDArray,
    err: npt.NDArray,
    elements: list[str],
    element_energies: dict[str, float],
) -> dict:
    """
    Fit the given spectrum with single gaussians using curve_fit.

    Args:
        energies: Array of energies
        counts: Array of counts
        elements: Elements to look for

    Returns:
        A dictionary containing the parameters of the detected lines and the
        fitted gaussian for each line.
    """

    params = {element: {} for element in elements}
    for element in elements:
        if element == "FeL":
            continue
        energy_window = [
            element_energies[element] - 0.125,
            element_energies[element] + 0.125,
        ]
        mask = np.where((energy_window[0] <= energies) & (energies <= energy_window[1]))
        energy_to_fit = energies[mask]
        counts_to_fit = counts[mask]
        a0 = np.max(counts_to_fit) - np.min([np.min(counts_to_fit), 0])
        mean0 = element_energies[element]
        sigma0 = 0.07
        try:
            popt, pcov = curve_fit(
                _single_gaussian,
                energy_to_fit,
                counts_to_fit,
                p0=[a0, mean0, sigma0],
                sigma=err[mask],
                absolute_sigma=True,
                maxfev=1000,
            )
            perr = np.sqrt(np.diag(pcov))
            params[element] = {
                "amplitude": f"{popt[0]} +- {perr[0]}",
                "mean": f"{popt[1]} +- {perr[1]}",
                "stddev": f"{popt[2]} +- {perr[2]}",
                "fit_function": _single_gaussian(energies, *popt),
                "fit_status": "Success",
            }
        except Exception as e:
            params[element] = {
                "amplitude": f"a0 +- -1",
                "mean": f"mean0 +- -1",
                "stddev": f"sigma0 +- -1",
                "fit_function": -1,
                "fit_status": f"Failed with error: {e}",
            }
    if "FeL" in elements:
        if "O" not in elements or params["O"]["fit_status"] != "Success":
            params["FeL"] = {
                "amplitude": f"a0 +- -1",
                "mean": f"mean0 +- -1",
                "stddev": f"sigma0 +- -1",
                "fit_function": -1,
                "fit_status": "Failed with error: O not detected/fitted",
            }
        else:
            energy_window = [
                element_energies["FeL"] - 0.125,
                element_energies["FeL"] + 0.125,
            ]
            mask = np.where(
                (energy_window[0] <= energies) & (energies <= energy_window[1])
            )
            energy_to_fit = energies[mask]
            counts_to_fit = counts[mask]
            a0 = np.max(counts_to_fit) - np.min([np.min(counts_to_fit), 0])
            mean0 = element_energies["FeL"]
            sigma0 = 0.07
            counts_to_fit -= params["O"]["fit_function"][mask]
            try:
                popt, pcov = curve_fit(
                    _single_gaussian,
                    energy_to_fit,
                    counts_to_fit,
                    p0=[a0, mean0, sigma0],
                    sigma=err[mask],
                    absolute_sigma=True,
                    maxfev=1000,
                )
                perr = np.sqrt(np.diag(pcov))
                params["FeL"] = {
                    "amplitude": f"{popt[0]} +- {perr[0]}",
                    "mean": f"{popt[1]} +- {perr[1]}",
                    "stddev": f"{popt[2]} +- {perr[2]}",
                    "fit_function": _single_gaussian(energies, *popt),
                    "fit_status": "Success",
                }
            except Exception as e:
                params["FeL"] = {
                    "amplitude": f"a0 +- -1",
                    "mean": f"mean0 +- -1",
                    "stddev": f"sigma0 +- -1",
                    "fit_function": -1,
                    "fit_status": f"Failed with error: {e}",
                }

    return params


def fit_specutils(
    energies: npt.NDArray,
    counts: npt.NDArray,
    err: npt.NDArray,
    elements: list[str],
    element_energies: dict[str, float],
) -> dict:
    """
    Fit the given spectrum with single gaussians using specutils.

    Args:
        energies: Array of energies
        counts: Array of counts
        elements: Elements to look for

    Returns:
        A dictionary containing the parameters of the detected lines and the
        fitted gaussian for each line.
    """
    params = {element: {} for element in elements}
    elems = elements.copy()
    flag = "FeL" in elements
    if flag:
        elems.remove("FeL")
        if "O" not in elements:
            params["FeL"] = {
                "amplitude": f"a0 +- -1",
                "mean": f"mean0 +- -1",
                "stddev": f"sigma0 +- -1",
                "fit_function": -1,
                "fit_status": "Failed with error: O not detected",
            }
            flag = False

    spectrum = Spectrum1D(
        flux=counts * u.count,  # type: ignore
        spectral_axis=energies * u.keV,  # type: ignore
        uncertainty=StdDevUncertainty(err * u.count),  # type: ignore
    )

    initial_gauss = []
    for element in elems:
        energy_window = [
            element_energies[element] - 0.125,
            element_energies[element] + 0.125,
        ]
        mask = np.where((energy_window[0] <= energies) & (energies <= energy_window[1]))
        counts_to_fit = counts[mask]

        a0 = np.max(counts_to_fit) - np.min([np.min(counts_to_fit), 0])
        mean0 = element_energies[element]
        sigma0 = 0.07
        initial_gauss.append(
            models.Gaussian1D(
                amplitude=a0 * u.count,  # type: ignore
                mean=mean0 * u.keV,  # type: ignore
                stddev=sigma0 * u.keV,  # type: ignore
            )
        )
    try:
        gauss_params = fit_lines(spectrum, initial_gauss, window=0.125 * u.keV, get_fit_info=True)  # type: ignore
        fit_status = "Success"
    except Exception as e:
        gauss_params = initial_gauss
        fit_status = f"Failed with error: {e}"

    for gauss, element in zip(gauss_params, elems):
        if fit_status == "Success":
            y_fit = gauss(energies * u.keV)  # type: ignore
            perr = np.sqrt(np.diag(gauss.meta["fit_info"]["param_cov"]))
            params[element] = {
                "amplitude": f"{gauss.amplitude.value} +- {perr[0]}",
                "mean": f"{gauss.mean.value} +- {perr[1]}",
                "stddev": f"{gauss.stddev.value} +- {perr[2]}",
                "fit_function": y_fit,
                "fit_status": fit_status,
            }
        else:
            params[element] = {
                "amplitude": f"a0 +- -1",
                "mean": f"mean0 +- -1",
                "stddev": f"sigma0 +- -1",
                "fit_function": -1,
                "fit_status": fit_status,
            }

    if flag:
        if params["O"]["fit_status"] == "Success":
            spectrum_sub = Spectrum1D(
                flux=counts * u.count - params["O"]["fit_function"],  # type: ignore
                spectral_axis=energies * u.keV,  # type: ignore
                uncertainty=StdDevUncertainty(err * u.count),  # type: ignore
            )
            energy_window = [
                element_energies["FeL"] - 0.125,
                element_energies["FeL"] + 0.125,
            ]
            mask = np.where(
                (energy_window[0] <= energies) & (energies <= energy_window[1])
            )
            counts_to_fit = counts[mask]
            a0 = np.max(counts_to_fit) - np.min([np.min(counts_to_fit), 0])
            mean0 = element_energies["FeL"]
            sigma0 = 0.07
            initial_gauss = models.Gaussian1D(
                amplitude=a0 * u.count,  # type: ignore
                mean=mean0 * u.keV,  # type: ignore
                stddev=sigma0 * u.keV,  # type: ignore
            )
            try:
                gauss = fit_lines(spectrum_sub, initial_gauss, window=0.125 * u.keV, get_fit_info=True)  # type: ignore
                fit_status = "Success"
            except Exception as e:
                fit_status = f"Failed with error: {e}"

            if fit_status == "Success":
                y_fit = gauss(energies * u.keV)  # type: ignore
                perr = np.sqrt(np.diag(gauss.meta["fit_info"]["param_cov"]))  # type: ignore
                params["FeL"] = {
                    "amplitude": f"{gauss.amplitude.value} +- {perr[0]}",  # type: ignore
                    "mean": f"{gauss.mean.value} +- {perr[1]}",  # type: ignore
                    "stddev": f"{gauss.stddev.value} +- {perr[2]}",  # type: ignore
                    "fit_function": y_fit,
                    "fit_status": fit_status,
                }
            else:
                params["FeL"] = {
                    "amplitude": f"a0 +- -1",
                    "mean": f"mean0 +- -1",
                    "stddev": f"sigma0 +- -1",
                    "fit_function": -1,
                    "fit_status": fit_status,
                }
        else:
            params["FeL"] = {
                "amplitude": f"a0 +- -1",
                "mean": f"mean0 +- -1",
                "stddev": f"sigma0 +- -1",
                "fit_function": -1,
                "fit_status": "Failed with error: O not detected",
            }
    return params
