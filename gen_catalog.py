import argparse
import json
import os
import re
import warnings

import numpy as np
import tqdm

warnings.filterwarnings("ignore")

from utils import display_text, parse_toml_params


def apply_cut(
    data: dict,
    elements: list[str],
    strict_elements: list[str],
    element_energies: dict[str, float],
) -> dict[str, list[tuple]]:
    """
    Find the valid Julian Dates for each element in the data.
    """
    valid_jds: dict[str, list] = {element: [] for element in elements}
    for element in elements:
        bkg_amp_std: float = data[element]["bkg_amp_std"]
        if data[element].get("threshold") is None:
            continue
        threshold: float = data[element]["threshold"]
        for jd, params in data[element].items():
            if not isinstance(params, dict):
                continue
            if params["fit_status"] != "Success":
                continue
            amp, err_amp = (float(x) for x in params["amplitude"].split("+-"))
            stddev, err_stddev = (float(x) for x in params["stddev"].split("+-"))
            mean, _ = (float(x) for x in params["mean"].split("+-"))
            flux = amp * stddev * np.sqrt(2 * np.pi)
            err_flux = np.sqrt((err_amp / amp) ** 2 + (err_stddev / stddev) ** 2) * flux
            if element in strict_elements:
                if (
                    amp > threshold * bkg_amp_std
                    and amp > 0
                    and np.abs(mean - element_energies[element]) < 0.05
                    and stddev < 0.1
                    and stddev > 0.05
                ):
                    valid_jds[element].append((jd, flux, err_flux))
            else:
                if (
                    amp > 0
                    and stddev > 0.05
                    and stddev < 0.15
                    and np.abs(mean - element_energies[element]) < 0.05
                ):
                    valid_jds[element].append((jd, flux, err_flux))
    return valid_jds


if __name__ == "__main__":
    params = parse_toml_params("config.toml")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datadir", type=str, required=True, default=params["gen_catalog"]["datadir"]
    )
    parser.add_argument(
        "--resultsdir",
        type=str,
        required=True,
        default=params["gen_catalog"]["resultsdir"],
    )
    parser.add_argument(
        "--years",
        nargs="+",
        type=str,
        default=["2019", "2020", "2021", "2022", "2023", "2024", "2025"],
    )
    args = parser.parse_args()

    os.makedirs(args.resultsdir, exist_ok=True)
    strict_elements: list[str] = params["elements"]["strict"]
    element_energies: dict[str, float] = params["elements"]["energies"]
    elements = list(element_energies.keys())

    catalog = {}
    for year in args.years:
        display_text(f"Processing {year}", "green")
        for month in tqdm.tqdm(sorted(os.listdir(f"{args.datadir}/{year}"))):
            for entry in sorted(os.listdir(f"{args.datadir}/{year}/{month}/")):
                if not re.match(r"\d{4}-\d{2}-\d{2}.json", entry):
                    continue

                file = f"{args.datadir}/{year}/{month}/{entry}"
                temp_catalog = {}
                with open(file, "r") as f:
                    data = json.load(f)
                    valid_jds_elemwise = apply_cut(
                        data, elements, strict_elements, element_energies
                    )

                    for element, detections in valid_jds_elemwise.items():
                        for jd, flux, err_flux in detections:
                            if jd not in temp_catalog:
                                temp_catalog[jd] = {}
                            temp_catalog[jd][element] = {
                                "flux": flux,
                                "err_flux": err_flux,
                            }

                    for jd in temp_catalog.keys():
                        coords = data["coords"][jd]
                        lat = np.mean(
                            [
                                np.mean(coords["V0_LAT"]),
                                np.mean(coords["V1_LAT"]),
                                np.mean(coords["V2_LAT"]),
                                np.mean(coords["V3_LAT"]),
                            ]
                        )
                        lon = np.mean(
                            [
                                np.mean(coords["V0_LON"]),
                                np.mean(coords["V1_LON"]),
                                np.mean(coords["V2_LON"]),
                                np.mean(coords["V3_LON"]),
                            ]
                        )
                        temp_catalog[jd]["lat"] = lat
                        temp_catalog[jd]["lon"] = lon
                catalog.update(temp_catalog)

    with open(f"{args.resultsdir}/catalog.json", "w") as f:
        json.dump(catalog, f, indent=4)
