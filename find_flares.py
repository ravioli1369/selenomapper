import argparse
import json
import os
import shutil
import time
from glob import glob
from multiprocessing import Pool
import gc

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import tqdm
from astropy.coordinates import get_body, get_sun
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.time import Time, TimeDelta
import warnings

warnings.filterwarnings("ignore")

from utils import parse_toml_params, detect_lines, display_text

mlp_params = {
    "figure.figsize": [9, 6],
    "axes.labelsize": 22,
    "axes.titlesize": 24,
    "axes.titlepad": 15,
    "figure.titlesize": 24,
    "axes.labelpad": 10,
    "font.size": 16,
    "legend.fontsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "text.usetex": True if shutil.which("latex") else False,
    "font.family": "serif",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.top": True,
    "ytick.left": True,
    "ytick.right": True,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.minor.size": 2.5,
    "xtick.major.size": 5,
    "ytick.minor.size": 2.5,
    "ytick.major.size": 5,
    "axes.axisbelow": True,
    "figure.dpi": 200,
}
plt.rcParams.update(mlp_params)


class FindFlares:
    """
    Class to find flares in the given data.
    """

    def __init__(
        self,
        timebin: float,
        datadir: str,
        resultsdir: str,
        animate: bool,
        fit_func: str,
        N_sigma: bool,
        params: dict,
        bkg_masterfiles: dict,
    ) -> None:
        """
        Initialize the FindFlares class.

        Args:
            timebin: Time bin for the data
            datadir: Directory containing the data
            resultsdir: Directory to save the results
            animate: Whether to animate the light curve
            fit_func: Fitting function to use, choose between 'curve_fit' and 'specutils'
            N_sigma: Whether to use Nsigma threshold for detection
            params: Dictionary containing the parameters from `config.toml`
            bkg_masterfiles: dictionary of phasewise background masterfiles

        Returns:
            None
        """

        self.datadir = datadir
        self.resultsdir = resultsdir
        os.makedirs(self.resultsdir, exist_ok=True)
        self.delta_t = timebin
        self.delta_index: int = int(self.delta_t // 8)  # each spectra is 8 seconds long
        self.animate = animate
        self.N_sigma = N_sigma
        self.threshold = 5
        self.bkg_masterfiles = bkg_masterfiles

        self.element_energies = params["elements"]["energies"]
        self.elements = list(self.element_energies.keys())
        self.energies = np.arange(0, 2048) * 13.5 / 1000

        self.good_energy_channel_range = [
            self.energy2channel(0.8),
            self.energy2channel(10.8),
        ]
        self.fit_func = fit_func

    def sort_by_time(
        self, data: npt.NDArray, headers: npt.NDArray
    ) -> tuple[npt.NDArray, tuple[npt.NDArray, npt.NDArray]]:
        """
        Sort the given FITS data and headers by the MID_UTC keyword in the header.

        Args:
            data: Array of fits data
            headers: Array of fits headers

        Returns:
            A tuple containing the sorted times and a tuple of sorted data, headers.
            The times are sorted in ascending order and are a numpy array of Julian
            Dates.
        """
        times = []
        for header in headers:
            times.append(header["MID_UTC"])
        times = np.array(Time(times).jd)
        sort = times.argsort()
        # hduls is a tuple of data and headers instead of the conventional HDUList
        hduls = (data[sort], headers[sort])
        return times[sort], hduls

    def combine_spectra(
        self,
        mask: npt.NDArray,
        times: npt.NDArray,
        hduls: tuple[npt.NDArray, npt.NDArray],
    ) -> tuple[npt.NDArray, Time, npt.NDArray, dict, list]:
        """
        Combine the spectra in the given time range.

        Args:
            mask: Mask of the indices of the spectra to combine.
            times: Array of JDs of all the spectra. This should be sorted before
                passing to this function
            hduls: Tuple of data and headers of all the spectra. This should be
                sorted by time before passing to this function

        Returns:
            A tuple containing the mean counts, the header of the combined
            data, and a 2D array of counts of each spectra that was combined
            and the (lat, long) coordinates of each spectra that was combined
            and the altitude of the satellite for each spectra that was combined
        """

        mid_jd = np.mean(times[mask])
        mid_utc = Time(mid_jd, format="jd")

        data, header = hduls
        data_to_combine, header_to_combine = data[mask], header[mask]
        mean_combined_data = np.copy(data_to_combine[0], order="C")

        all_coords = {}
        all_altitudes = []
        for i in range(4):
            all_coords[f"V{i}_LAT"] = []
            all_coords[f"V{i}_LON"] = []
        counts = []
        for _data, _header in zip(data_to_combine, header_to_combine):
            counts.append(np.array(_data["COUNTS"]).astype(float))
            all_altitudes.append(_header["SAT_ALT"])
            for i in range(4):
                all_coords[f"V{i}_LAT"].append(_header[f"V{i}_LAT"])
                all_coords[f"V{i}_LON"].append(_header[f"V{i}_LON"])

        counts = np.array(counts)
        mean_combined_counts = np.mean(counts, axis=0)
        mean_combined_data["COUNTS"] = mean_combined_counts
        return mean_combined_counts, mid_utc, counts, all_coords, all_altitudes

    def get_spectra_files(
        self, year: str, month: str, day: str
    ) -> tuple[npt.NDArray, tuple[npt.NDArray, npt.NDArray]]:
        """
        Get the light (spectra taken for the sunlit side of moon) or dark (spectra
        taken for the non sunlit side of moon) files for the given date.

        Args:
            year: Year of the date
            month: Month of the date
            day: Day of the date

        Returns:
            A tuple containing the times, array of data, and headers of the spectra
        """
        data, headers = [], []
        light_directory = f"{self.datadir}/{year}/{month}/{day}/light"

        files = glob(f"{light_directory}/*.fits")
        for file in files:
            data.append(fits.getdata(file, memmap=False))
            headers.append(dict(fits.getheader(file, 1, memmap=False)))

        data, headers = np.array(data), np.array(headers)

        # hduls is a tuple of data and headers instead of the conventional HDUList
        times, hduls = self.sort_by_time(data, headers)
        return times, hduls

    def energy2channel(self, energy: float) -> int:
        """
        Convert the given energy to the corresponding channel number.

        Args:
            energy: Energy in keV

        Returns:
            The channel number corresponding to the given energy
        """
        return int(energy * 1000 / 13.5)  # keV to eV then divide by gain = 13.5 eV

    def moon_phase_angle(self, time) -> tuple[float, float]:
        """
        Calculate lunar orbital phase and elongation of the moon.

        Args:
            time: Time of observation in astropy.time.Time format

        Returns:
            phase_ang_rad: Lunar orbital phase in radians
            elong_deg: Elongation of the moon in degrees

        """

        sun = get_sun(time)
        moon = get_body("moon", time)
        elongation = sun.separation(moon)
        elong_deg = elongation.deg
        phase_ang_rad = np.arctan2(
            sun.distance * np.sin(elongation),
            moon.distance - sun.distance * np.cos(elongation),
        )
        return phase_ang_rad, float(elong_deg)  # type: ignore

    def check_elong(self, time) -> str:
        """
        Check if the moon is in first or third quarter.

        Args:
            time: Time of observation in astropy.time.Time format

        Returns:
            phase: str
        """
        time_next = time + TimeDelta(1, format="jd")
        _, elongation = self.moon_phase_angle(time)
        _, elongation_next = self.moon_phase_angle(time_next)
        if elongation_next > elongation:
            return "first_quarter"
        return "third_quarter"

    def elongation_to_phase(self, time) -> str:
        """
        Calculate the phase of the moon from its elongation.

        Args:
            time: Time of observation in astropy.time.Time

        Returns:
            phase: str
        """

        _, elongation_deg = self.moon_phase_angle(time)

        if elongation_deg >= 0 and elongation_deg <= 45:
            return "new_moon"
        elif elongation_deg > 45 and elongation_deg <= 135:
            return self.check_elong(time)
        else:
            return "full_moon"

    def get_mean_bkg(self, year: str, month: str, day: str) -> npt.NDArray:
        """
        Get the sigma clipped mean background counts for the given date.

        Args:
            year: Year of the date
            month: Month of the date
            day: Day of the date

        Returns:
            The clipped mean background counts
        """
        time_ = Time(f"{year}-{month}-{day}") if year != "2019" else Time(f"2019-12-31")
        time_for_phase = Time(f"{year}-{month}-{day}")
        one_year_ago = time_ - TimeDelta(365, format="jd")
        phase = self.elongation_to_phase(time_for_phase)

        self.bkg_counts = self.bkg_masterfiles[phase]["bkg_counts"]
        bkg_jds = self.bkg_masterfiles[phase]["JD"]

        split_index_year_back = np.searchsorted(  # type: ignore
            bkg_jds, float(one_year_ago.jd), side="right"  # type: ignore
        )
        split_index_current_time = np.searchsorted(  # type: ignore
            bkg_jds, float(time_.jd), side="right"  # type: ignore
        )

        self.bkg_counts = self.bkg_counts[
            split_index_year_back:split_index_current_time
        ]

        filtered_bkg = self.bkg_counts[
            np.sum(self.bkg_counts[:, 37:801], axis=1) < 2000
        ]
        self.N_bkg = len(filtered_bkg)
        mean_bkg_counts = np.mean(filtered_bkg, axis=0)
        return mean_bkg_counts

    def calculate_sigma(
        self, bkg_counts: npt.NDArray | None
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        Calculate the standard deviation of the background counts in the energy
        range of each element.

        Args:
            bkg_counts: 2D array of background counts. Size should be (len(bkg_times), N_channels)

        Returns:
            A tuple of dictionaries containing the standard deviation of the flux of each
            element and the standard deviation of the
            amplitude of each element in the background counts
        """
        bkg_counts = self.bkg_counts if bkg_counts is None else bkg_counts

        elem_bkg_counts_dist = {element: [] for element in self.elements}
        for i in range(0, len(bkg_counts) - 1, self.delta_index):
            j = (
                (i + self.delta_index)
                if (i + self.delta_index) < len(bkg_counts)
                else -1
            )
            for element in self.elements:
                min_channel = self.energy2channel(
                    self.element_energies[element] - 0.125
                )
                max_channel = self.energy2channel(
                    self.element_energies[element] + 0.125
                )

                counts = np.mean(bkg_counts[i:j, min_channel : max_channel + 1], axis=0)
                elem_bkg_counts_dist[element].append(counts)

        elem_bkg_flux_dist = {
            element: np.sum(elem_bkg_counts_dist[element], axis=1)
            for element in self.elements
        }

        elem_bkg_flux_std_devs = {
            element: float(sigma_clipped_stats(elem_bkg_flux_dist[element])[2])
            for element in self.elements
        }
        elem_bkg_amp_std_devs = {
            element: float(
                np.mean(sigma_clipped_stats(elem_bkg_counts_dist[element], axis=1)[2])
            )
            for element in self.elements
        }  # this is equivalent to elem_bkg_flux_std_devs / sqrt(max_channel + 1 - min_channel)
        return elem_bkg_flux_std_devs, elem_bkg_amp_std_devs

    def process_bkg(self, year: str, month: str, day: str) -> None:
        """
        Process the background data for the given date. Creates the mean
        background and calculates the standard deviations of the background.

        Args:
            year: Year of the date
            month: Month of the date
            day: Day of the date

        Returns:
            None
        """
        self.mean_bkg_counts = self.get_mean_bkg(year, month, day)
        self.elem_bkg_flux_std_devs, self.elem_bkg_amp_std_devs = self.calculate_sigma(
            self.bkg_counts
        )

    def find_flares(self, year: str, month: str, day: str) -> None:
        """
        Find flares in the given date.

        Args:
            year: Year of the date
            month: Month of the date
            day: Day of the date

        Returns:
            None
        """
        # get the mean background counts and the standard deviation of the background
        self.process_bkg(year, month, day)

        detections: dict[str, dict] = {
            element: {
                "bkg_flux_std": self.elem_bkg_flux_std_devs[element],
                "bkg_amp_std": self.elem_bkg_amp_std_devs[element],
            }
            for element in self.elements
        }
        detections["coords"] = {}
        detections["altitude"] = {}
        light_times, light_hduls = self.get_spectra_files(year, month, day)

        light_curve_y = {element: [] for element in self.elements}
        light_curve_x = []

        (
            mean_combined_light_counts_list,
            light_err_list,
            all_coords_list,
            all_altitudes_list,
        ) = ([], [], [], [])

        for i in range(0, len(light_times), self.delta_index):
            light_time = light_times[i]
            start_time = light_time
            end_time = Time(start_time, format="jd") + TimeDelta(
                self.delta_t, format="sec"
            )
            _start_time = Time(start_time, format="jd")
            _end_time = Time(end_time, format="jd")
            times = Time(light_times, format="jd")  # type: ignore
            mask = np.where((times >= _start_time) & (times < _end_time))[0]

            if not len(mask) >= self.delta_index:  # don't do rolling average
                # at the end of an orbit as there are not enough spectra to create
                # the contiuum profile and bkg subtraction will not be accurate
                continue

            (
                mean_combined_light_counts,
                mid_utc,
                light_counts,
                all_coords,
                all_altitudes,
            ) = self.combine_spectra(
                mask,
                light_times,
                light_hduls,
            )
            all_coords_list.append(all_coords)
            all_altitudes_list.append(all_altitudes)
            N_light = len(light_counts)
            err_light = np.sqrt(
                mean_combined_light_counts / N_light + self.mean_bkg_counts / self.N_bkg
            )
            mean_combined_light_counts -= self.mean_bkg_counts
            mean_combined_light_counts -= np.median(
                mean_combined_light_counts[
                    self.good_energy_channel_range[0] : self.good_energy_channel_range[
                        1
                    ]
                ]
            )
            light_curve_x.append(mid_utc.jd)

            mean_combined_light_counts_list.append(mean_combined_light_counts)
            light_err_list.append(err_light)

            elem_amps = {element: 0.0 for element in self.elements}
            elem_fluxes = {element: 0.0 for element in self.elements}
            for element in self.elements:
                min_channel = self.energy2channel(
                    self.element_energies[element] - 0.125
                )
                max_channel = self.energy2channel(
                    self.element_energies[element] + 0.125
                )
                elem_fluxes[element] = float(
                    np.sum(mean_combined_light_counts[min_channel : max_channel + 1])
                )
                elem_amps[element] = float(
                    np.max(mean_combined_light_counts[min_channel : max_channel + 1])
                    - np.min(
                        [
                            float(
                                np.min(
                                    mean_combined_light_counts[
                                        min_channel : max_channel + 1
                                    ]
                                )
                            ),
                            0,
                        ]
                    )
                )
                light_curve_y[element].append(
                    elem_fluxes[element] / self.elem_bkg_flux_std_devs[element]
                )

        all_coords_array = np.array(all_coords_list)
        light_curve_x = np.array(light_curve_x)
        light_curve_y_array = np.array(
            [light_curve_y[element] for element in self.elements]
        )

        self.threshold = 5
        if self.N_sigma:
            _, med, std = sigma_clipped_stats(light_curve_y_array, axis=1)
            thres = min(med + 5 * std)
            self.threshold = min(thres, self.threshold)

        flag = np.max(light_curve_y_array, axis=0) > self.threshold
        flag_elem = light_curve_y_array > self.threshold
        elems = np.array(
            [
                np.array(self.elements)[flag_elem[:, i]].tolist()
                for i in range(len(flag))
            ],
            dtype=object,
        )
        
        with Pool(8) as p:
            params = p.starmap(
                detect_lines,
                [
                    (
                        np.array(counts),
                        np.array(err),
                        self.energies,
                        elem,
                        self.element_energies,
                        self.fit_func,
                    )
                    for counts, err, elem in zip(
                        mean_combined_light_counts_list, light_err_list, elems
                    )
                ],
            )


        if self.animate:
            step = self.delta_index
            fig = plt.figure(figsize=(10, 7))
            ax = plt.axes(xlim=(0, 7), ylim=(-20, 400), xlabel="Energies (keV)", ylabel="Counts", title=f"{year}-{month}-{day}")
            
            num_plts = len(self.elements) + 1
            cmap = plt.get_cmap("tab20")
            plts = []
            energies = np.arange(2048) * 13.5 / 1000
            mask = np.where((energies <= 7))
            energies = energies[mask]
            
            def init():
                plts.append(ax.plot([], [], color=cmap(0), label="Spectrum", alpha=0.6)[0])
                
                for i, element in enumerate(self.elements):
                    plts.append(ax.plot([], [], color=cmap(i+1), label=element)[0])
                ax.legend(loc="upper right")
                return plts
            
            def animate(i):
                plts[0].set_data(energies, mean_combined_light_counts_list[step * i][mask])
                for j, element in enumerate(self.elements):
                    if params[step*i] is None or element not in elems[step*i] or params[step*i][element]["fit_status"] != "Success":
                        plts[j+1].set_data([], [])
                    else:
                        plts[j+1].set_data(energies, params[step*i][element]["fit_function"][mask])
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend(handles[:num_plts], labels[:num_plts], loc="upper right")
                return plts
            
            ani = animation.FuncAnimation(
                fig, animate, init_func=init, frames=len(mean_combined_light_counts_list) // step 
            )
            ani.save(
                f"{self.resultsdir}/{year}-{month}-{day}.mp4",
                writer=animation.FFMpegWriter(fps=10),
            )
            del ani
            gc.collect()
            plt.close()
        
        for i, (param, mid_jd, coords, altitude, elem) in enumerate(
            zip(params, light_curve_x, all_coords_array, all_altitudes_list, elems)
        ):
            if params == {}:
                continue
            detections["coords"][mid_jd] = coords
            detections["altitude"][mid_jd] = altitude
            for element in elem:
                param[element]["flux"] = (  # type: ignore
                    light_curve_y[element][i] * self.elem_bkg_flux_std_devs[element]
                )
                param[element].pop("fit_function")  # type: ignore
                detections[element]["threshold"] = self.threshold
                detections[element][mid_jd] = param[element]  # type: ignore

        with open(
            f"{self.resultsdir}/{year}-{month}-{day}.json", "w", encoding="utf-8"
        ) as f:
            json.dump(detections, f, indent=4)

        fig = plt.figure(figsize=(10, 7))
        cmap = plt.get_cmap("tab20")
        for elem_idx, element in enumerate(self.elements):
            plt.plot([], [], ".", ms=15, label=element, color=cmap(elem_idx))
            plt.scatter(
                light_curve_x,
                light_curve_y[element],
                s=5,
                alpha=0.4,
                color=cmap(elem_idx),
            )
        plt.axhline(self.threshold, color="black", linestyle="--", label="Nsigma")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.xlabel("Time (JD)")
        plt.ylabel(r"Number of $\sigma$ above background")
        plt.title(f"{year}-{month}-{day}")
        fig.tight_layout()
        fig.savefig(f"{self.resultsdir}/{year}-{month}-{day}.png", bbox_inches="tight")
        plt.close()


if __name__ == "__main__":
    params = parse_toml_params("config.toml")

    parser = argparse.ArgumentParser(
        description="Find flares in the given data using curve fitting.\
        The `timebin`, `datadir`, `resultsdir`, `fit_func`, `Nsigma`\
        arguments can be passed in the `config.toml` or through the parser."
    )
    parser.add_argument(
        "--year", type=str, default=None, help="Year of the date to process"
    )
    parser.add_argument(
        "--month", type=str, default=None, help="Month of the date to process"
    )
    parser.add_argument(
        "--day", type=str, default=None, help="Day of the date to process"
    )
    parser.add_argument(
        "--runall", action="store_true", help="Run for all the days in a month or year"
    )
    parser.add_argument(
        "--timebin",
        type=float,
        default=params["find_flares"]["timebin"],
        help="Time bin for detection",
    )
    parser.add_argument(
        "--datadir",
        type=str,
        default=params["find_flares"]["datadir"],
        help="Directory containing the data",
    )
    parser.add_argument(
        "--resultsdir",
        type=str,
        default=params["find_flares"]["resultsdir"],
        help="Directory to save the results",
    )
    parser.add_argument(
        "--Nsigma",
        type=bool,
        default=params["find_flares"]["Nsigma"],
        help="Whether to use Nsigma detection threshold",
    )
    parser.add_argument("--animate", action="store_true", help="Animate the spectra")
    parser.add_argument(
        "--fit_func",
        type=str,
        default=params["find_flares"]["fit_func"],
        help="Fitting function to use",
    )
    assert parser.parse_args().fit_func in [
        "curve_fit",
        "specutils",
    ], "Invalid fit function"
    args = parser.parse_args()

    def runall(resultsdir, bkg_masterfile_dict):
        """
        Dummy function to be called by the Pool.map function to run the find_flares
        in parallel for all the days in the given directory.

        Args:
            resultsdir: Directory containing the results of the day

        Returns:
            None
        """
        year = resultsdir.split("/")[-3]
        month = resultsdir.split("/")[-2]
        day = resultsdir.split("/")[-1]
        resultsdir = os.path.dirname(resultsdir)

        _detector = FindFlares(
            args.timebin,
            args.datadir,
            resultsdir,
            args.animate,
            args.fit_func,
            args.Nsigma,
            params,
            bkg_masterfiles=bkg_masterfile_dict,
        )
        try:
            _detector.find_flares(year, month, day)
        except Exception as e:
            file = f"{resultsdir}/{day}.txt"
            with open(file, "a") as f:
                f.write(f"{e}\n")
                f.close()
        
        del _detector
        gc.collect()

    bkg_masterfile_dict = {}

    for phase in ["new_moon", "full_moon", "first_quarter", "third_quarter"]:
        bkg_masterfile_dict[phase] = fits.getdata(
            f"{args.datadir}/{phase}_masterfile.fits", memmap=False
        )

    if args.runall:
        resultsdirs = []

        for _year in (
            sorted(filter(lambda x: x.isnumeric(), os.listdir(f"{args.datadir}/")))
            if args.year is None
            else [args.year]
        ):
            for _month in (
                sorted(
                    filter(
                        lambda x: x.isnumeric(), os.listdir(f"{args.datadir}/{_year}/")
                    )
                )
                if args.month is None
                else [args.month]
            ):
                for _day in sorted(
                    filter(
                        lambda x: x.isnumeric(),
                        os.listdir(f"{args.datadir}/{_year}/{_month}/"),
                    )
                ):
                    resultsdirs.append(f"{args.resultsdir}/{_year}/{_month}/{_day}")
        for resultsdir in tqdm.tqdm(resultsdirs):
            runall(resultsdir, bkg_masterfile_dict)
    else:
        start = time.time()

        detector = FindFlares(
            args.timebin,
            args.datadir,
            args.resultsdir,
            args.animate,
            args.fit_func,
            args.Nsigma,
            params,
            bkg_masterfiles=bkg_masterfile_dict,
        )
        detector.find_flares(args.year, args.month, args.day)
        display_text(
            f"Time taken to run {args.year}-{args.month}-{args.day} is {(time.time()-start):.2f} seconds",
            "green",
        )
