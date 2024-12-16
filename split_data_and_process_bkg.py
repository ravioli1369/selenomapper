import argparse
import glob
import os
import subprocess as subp
import time
from multiprocessing import Pool

import numpy as np
from astropy.coordinates import get_body, get_sun
from astropy.io import fits
from astropy.time import Time, TimeDelta
from utils import display_text, parse_toml_params
import warnings

warnings.filterwarnings("ignore")

MOON_RADIUS = 1737.5  # km


def separate_light_and_dark(file: str) -> None:
    """
    Move the file to the light or dark directory based on the solar angle
    and add a new column to the fits file with the energy values

    Args:
        file: Path to the fits file

    Returns:
        None
    """

    dirname = os.path.dirname(file)
    light_direc = f"{dirname}/light/"
    dark_direc = f"{dirname}/dark/"
    with fits.open(file, mode="update", memmap=False) as hdul:
        solar_angle = float(hdul[1].header["SOLARANG"])  # type: ignore
        energy = hdul[1].data["CHANNEL"] * 13.5 / 1000  # type: ignore

        # Create a new columns object with the new column and new hdu
        col = fits.Column(name="ENERGY", format="1E", array=energy)
        cols = hdul[1].columns + col  # type: ignore
        hdu = fits.BinTableHDU.from_columns(cols, header=hdul[1].header)  # type: ignore

        # Replace the old HDU with the new one and save the changes
        hdul[1] = hdu
        hdul.flush()
    os.makedirs(light_direc, exist_ok=True)
    os.makedirs(dark_direc, exist_ok=True)

    if solar_angle < 90:
        subp.call(f"mv {file} {light_direc}", shell=True)
    else:
        subp.call(f"mv {file} {dark_direc}", shell=True)


class OccultBkg:
    def __init__(self, datadir, year):
        self.datadir = datadir
        self.year = year

    def occult_condition(self, header_dict):
        return (
            1 / np.cos(np.deg2rad(header_dict["SOLARANG"] - 90)) - 1
        ) * MOON_RADIUS > header_dict["SAT_ALT"]

    def moon_phase_angle(self, time):
        """
        Calculate lunar orbital phase in radians using astropy data

        Args:
            time: Astropy Time object for time of observation

        Returns:
            Phase angle of the moon [radians], elongation from the sun [degree]
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

    def check_elong(self, file):
        """
        Check if the moon is increasing (first_quarter) or decreasing (third_quarter)
        in phase the next day

        Args:
            file: File name of the current day

        Returns:
            elongation: Elongation of the moon [radians]
        """
        # get the time from the file name
        time = Time(fits.getheader(file, 1, memmap=False)["MID_UTC"])
        time_next = time + TimeDelta(1, format="jd")
        _, elongation = self.moon_phase_angle(time)
        _, elongation_next = self.moon_phase_angle(time_next)
        if elongation_next > elongation:
            return "first_quarter"
        return "third_quarter"

    def elongation_to_phase(self, file):
        """
        Calculate the phase angle of the moon from its elongation.

        Args:
            file: FITS file containing the data. Only the MID_UTC header is used.

        Returns:
            phase: Phase angle of the moon [radians]
        """

        time = Time(fits.getheader(file, 1, memmap=False)["MID_UTC"])
        _, elongation_deg = self.moon_phase_angle(time)

        if elongation_deg >= 0 and elongation_deg <= 45:
            return "new_moon"
        elif elongation_deg > 45 and elongation_deg <= 135:
            return self.check_elong(file)
        elif elongation_deg > 135 and elongation_deg <= 180:
            return "full_moon"

    def phase_calc(self):
        """
        Calculate the phase of the moon for each day in the year and save the
        background counts along with JD for each phase to a fits file.
        """

        new_moon = []
        new_moon_jd = []

        first_quarter = []
        first_quarter_jd = []

        third_quarter = []
        third_quarter_jd = []

        full_moon = []
        full_moon_jd = []

        if not os.path.exists(f"{self.datadir}/{self.year}"):
            display_text(f"Year {self.year} not found", "red")
            return
        for month in sorted(
            filter(lambda x: x.isnumeric(), os.listdir(f"{self.datadir}/{self.year}/"))
        ):
            # for month in ['10']:

            display_text(f"Starting processing for {self.year}/{month}")
            month_dir = f"{self.datadir}/{self.year}/{month}"
            days_dir = glob.glob(f"{month_dir}/**/dark/*.fits", recursive=True)

            for file in days_dir:
                file_header = fits.getheader(file, 1, memmap=False)
                if self.occult_condition(file_header):
                    phase = self.elongation_to_phase(file)
                    counts = fits.getdata(file, memmap=False)["COUNTS"]  # type: ignore
                    try:
                        if phase == "new_moon":
                            new_moon_jd.append(Time(file_header["MID_UTC"]).jd)
                            new_moon.append(counts)
                        elif phase == "first_quarter":
                            first_quarter_jd.append(Time(file_header["MID_UTC"]).jd)
                            first_quarter.append(counts)
                        elif phase == "third_quarter":
                            third_quarter_jd.append(Time(file_header["MID_UTC"]).jd)
                            third_quarter.append(counts)
                        elif phase == "full_moon":
                            full_moon_jd.append(Time(file_header["MID_UTC"]).jd)
                            full_moon.append(counts)
                    except:
                        continue

        # saving the phase_dict to 4 different fits files according to the phases
        if len(new_moon) != 0:
            # Create the new moon masterfile
            columns = [
                fits.Column(name="JD", format="E", array=np.array(new_moon_jd)),
                fits.Column(
                    name="bkg_counts", format="2048E", array=np.array(new_moon)
                ),
            ]

            # Create the FITS table
            hdu = fits.BinTableHDU.from_columns(columns)

            hdu.writeto(
                f"{self.datadir}/{self.year}/new_moon_masterfile.fits", overwrite=True
            )

        if len(first_quarter) != 0:
            # Create the new moon masterfile
            columns = [
                fits.Column(name="JD", format="E", array=np.array(first_quarter_jd)),
                fits.Column(
                    name="bkg_counts", format="2048E", array=np.array(first_quarter)
                ),
            ]

            # Create the FITS table
            hdu = fits.BinTableHDU.from_columns(columns)
            hdu.writeto(
                f"{self.datadir}/{self.year}/first_quarter_masterfile.fits",
                overwrite=True,
            )

        if len(third_quarter) != 0:
            # Create the new moon masterfile
            columns = [
                fits.Column(name="JD", format="E", array=np.array(third_quarter_jd)),
                fits.Column(
                    name="bkg_counts", format="2048E", array=np.array(third_quarter)
                ),
            ]

            # Create the FITS table
            hdu = fits.BinTableHDU.from_columns(columns)
            hdu.writeto(
                f"{self.datadir}/{self.year}/third_quarter_masterfile.fits",
                overwrite=True,
            )

        if len(full_moon) != 0:
            # Create the new moon masterfile
            columns = [
                fits.Column(name="JD", format="E", array=np.array(full_moon_jd)),
                fits.Column(
                    name="bkg_counts", format="2048E", array=np.array(full_moon)
                ),
            ]

            # Create the FITS table
            hdu = fits.BinTableHDU.from_columns(columns)
            hdu.writeto(
                f"{self.datadir}/{self.year}/full_moon_masterfile.fits", overwrite=True
            )


def phases(tup):
    datadir, year = tup
    occultor = OccultBkg(datadir, year)
    occultor.phase_calc()


if __name__ == "__main__":

    params = parse_toml_params("config.toml")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datadir",
        help="Directory containing the files",
        default=params["split_data_and_process_bkg"]["datadir"],
    )
    parser.add_argument(
        "--year",
        help="Year to process",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-n",
        "--ncores",
        help="Number of cores to use",
        type=int,
        default=8,
    )

    args = parser.parse_args()
    years = (
        sorted(filter(lambda x: x.isnumeric(), os.listdir(f"{args.datadir}/")))
        if args.year is None
        else [args.year]
    )

    start = time.time()
    files = []
    for _year in years:
        files.extend(glob.glob(f"{args.datadir}/{_year}/*/*/*.fits", recursive=True))

    display_text("Separating light and dark files", "green")
    if len(files) == 0:
        display_text("light and dark already separated", "green")
    else:
        with Pool(args.ncores) as pool:
            pool.map(separate_light_and_dark, files)
        display_text(
            f"Separation of light and dark files done in {(time.time() - start):.2f}",
            "green",
        )
    with Pool(args.ncores) as pool:
        pool.map(phases, [(args.datadir, y) for y in years])

    # join all existing years into a masterfile
    # uses all existing years, not just provided year
    # make sure files exist for each year

    new_moon_combined_jd = []
    first_quarter_combined_jd = []
    third_quarter_combined_jd = []
    full_moon_combined_jd = []

    new_moon_combined_counts = []
    first_quarter_combined_counts = []
    third_quarter_combined_counts = []
    full_moon_combined_counts = []

    for _year in sorted(
        filter(lambda x: x.isnumeric(), os.listdir(f"{args.datadir}/"))
    ):
        # append the masterfiles
        new_moon = fits.open(
            f"{args.datadir}/{_year}/new_moon_masterfile.fits", memmap=False
        )
        first_quarter = fits.open(
            f"{args.datadir}/{_year}/first_quarter_masterfile.fits", memmap=False
        )
        third_quarter = fits.open(
            f"{args.datadir}/{_year}/third_quarter_masterfile.fits", memmap=False
        )
        full_moon = fits.open(
            f"{args.datadir}/{_year}/full_moon_masterfile.fits", memmap=False
        )

        new_moon_combined_jd.append(new_moon[1].data["JD"])
        first_quarter_combined_jd.append(first_quarter[1].data["JD"])
        third_quarter_combined_jd.append(third_quarter[1].data["JD"])
        full_moon_combined_jd.append(full_moon[1].data["JD"])

        new_moon_combined_counts.append(new_moon[1].data["bkg_counts"])
        first_quarter_combined_counts.append(first_quarter[1].data["bkg_counts"])
        third_quarter_combined_counts.append(third_quarter[1].data["bkg_counts"])
        full_moon_combined_counts.append(full_moon[1].data["bkg_counts"])

    new_moon_combined_jd = np.concatenate(new_moon_combined_jd)
    first_quarter_combined_jd = np.concatenate(first_quarter_combined_jd)
    third_quarter_combined_jd = np.concatenate(third_quarter_combined_jd)
    full_moon_combined_jd = np.concatenate(full_moon_combined_jd)

    new_moon_combined_counts = np.concatenate(new_moon_combined_counts)
    first_quarter_combined_counts = np.concatenate(first_quarter_combined_counts)
    third_quarter_combined_counts = np.concatenate(third_quarter_combined_counts)
    full_moon_combined_counts = np.concatenate(full_moon_combined_counts)

    new_moon_columns = [
        fits.Column(name="JD", format="E", array=new_moon_combined_jd),
        fits.Column(name="bkg_counts", format="2048E", array=new_moon_combined_counts),
    ]
    # Create the new FITS table HDU
    new_moon_hdu = fits.BinTableHDU.from_columns(new_moon_columns)
    # Write the combined data to a new FITS file
    output_file = f"{args.datadir}/new_moon_masterfile.fits"
    new_moon_hdu.writeto(output_file, overwrite=True)

    first_quarter_columns = [
        fits.Column(name="JD", format="E", array=first_quarter_combined_jd),
        fits.Column(
            name="bkg_counts", format="2048E", array=first_quarter_combined_counts
        ),
    ]
    # Create the new FITS table HDU
    first_quarter_hdu = fits.BinTableHDU.from_columns(first_quarter_columns)
    # Write the combined data to a new FITS file
    output_file = f"{args.datadir}/first_quarter_masterfile.fits"
    first_quarter_hdu.writeto(output_file, overwrite=True)

    third_quarter_columns = [
        fits.Column(name="JD", format="E", array=third_quarter_combined_jd),
        fits.Column(
            name="bkg_counts", format="2048E", array=third_quarter_combined_counts
        ),
    ]

    # Create the new FITS table HDU

    third_quarter_hdu = fits.BinTableHDU.from_columns(third_quarter_columns)
    # Write the combined data to a new FITS file
    output_file = f"{args.datadir}/third_quarter_masterfile.fits"
    third_quarter_hdu.writeto(output_file, overwrite=True)

    full_moon_columns = [
        fits.Column(name="JD", format="E", array=full_moon_combined_jd),
        fits.Column(name="bkg_counts", format="2048E", array=full_moon_combined_counts),
    ]
    # Create the new FITS table HDU
    full_moon_hdu = fits.BinTableHDU.from_columns(full_moon_columns)
    # Write the combined data to a new FITS file
    output_file = f"{args.datadir}/full_moon_masterfile.fits"
    full_moon_hdu.writeto(output_file, overwrite=True)

    display_text(f"Total time taken {(time.time() - start):.2f}")
