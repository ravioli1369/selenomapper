import argparse
import gc
import json
import os
import re
import warnings
from multiprocessing import Pool

import numpy as np
import numpy.typing as npt
import rasterio
import tqdm
from rasterio.enums import MergeAlg
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.ops import split, unary_union

from utils import parse_toml_params

warnings.filterwarnings("ignore")

min_lon, max_lon = -180, 180
min_lat, max_lat = -90, 90


def split_poly(
    ULLon: float,
    ULLat: float,
    LLLon: float,
    LLLat: float,
    LRLon: float,
    LRLat: float,
    URLon: float,
    URLat: float,
) -> tuple[Polygon, Polygon]:
    """
    Splits a polygon across the 180° longitude line and adjusts coordinates to
    fit within the [-180, 180] range.

    Args:
        ULLon (float): Upper left longitude.
        ULLat (float): Upper left latitude.
        LLLon (float): Lower left longitude.
        LLLat (float): Lower left latitude.
        LRLon (float): Lower right longitude.
        LRLat (float): Lower right latitude.
        URLon (float): Upper right longitude.
        URLat (float): Upper right latitude.

    Returns:
        Tuple[Optional[Polygon], Optional[Polygon]]: Two polygons representing
        the split parts if polygon crosses 180°; otherwise, returns `None` for
        each part.
    """
    ULLon = 360 + ULLon if ULLon < 0 else ULLon
    URLon = 360 + URLon if URLon < 0 else URLon
    LRLon = 360 + LRLon if LRLon < 0 else LRLon
    LLLon = 360 + LLLon if LLLon < 0 else LLLon
    pol = Polygon(
        [[ULLon, ULLat], [LLLon, LLLat], [LRLon, LRLat], [URLon, URLat], [ULLon, ULLat]]
    )
    line = LineString([(180, 90), (180, -90)])
    spl_pol = split(pol, line)
    poly1, poly2 = None, None
    for pol in spl_pol.geoms:
        if True in [item > 180 for item in pol.exterior.coords.xy[0]]:  # type: ignore
            new_lon = [
                item - 360 if item >= 180 else item
                for item in pol.exterior.coords.xy[0]  # type: ignore
            ]
            lat = pol.exterior.coords.xy[1]  # type: ignore
            poly1 = Polygon(zip(*(new_lon, lat)))
        else:
            poly2 = pol
    return poly1, poly2  # type: ignore


def calculate_central_square(
    coords: dict, moon_radius=1737.5, side_of_square=12.5
) -> dict:
    """
    Calculates the coordinates of the central square of the detection region.

    Args:
        coords (dict): Dictionary containing the coordinates of the detection region
        moon_radius (float, optional): Radius of the moon in km. Defaults to 1737.5.
        side_of_square (float, optional): Side of the square in km. Defaults to 12.5.

    Returns:
        dict: Dictionary containing the coordinates of the central square of the
        detection region.
    """
    n = len(coords["V0_LON"])
    square_coords = {}
    center_long = np.mean(
        [
            [
                coords["V0_LON"][i],
                coords["V1_LON"][i],
                coords["V2_LON"][i],
                coords["V3_LON"][i],
            ]
            for i in range(n)
        ],
        axis=1,
    )
    center_lat = np.mean(
        [
            [
                coords["V0_LAT"][i],
                coords["V1_LAT"][i],
                coords["V2_LAT"][i],
                coords["V3_LAT"][i],
            ]
            for i in range(n)
        ],
        axis=1,
    )
    for i in range(4):
        relative_direction = (
            coords[f"V{i}_LON"] - center_long > 0,
            coords[f"V{i}_LAT"] - center_lat > 0,
        )
        square_coords[f"V{i}_LON"] = center_long + (
            side_of_square * 180 / (np.pi * moon_radius)
        ) * (2 * relative_direction[0] - 1)
        square_coords[f"V{i}_LAT"] = center_lat + (
            side_of_square * 180 / (np.pi * moon_radius)
        ) * (2 * relative_direction[1] - 1)
    return square_coords


def create_polygon(coords: dict) -> Polygon | MultiPolygon:
    """
    Creates a polygon from four corner coordinates, splitting across 180°
    longitude if necessary.

    Args:
        coords (dict): Dictionary with corner coordinates, expected to contain
        keys "V0_LON", "V0_LAT", etc.

    Returns:
        Union[Polygon, MultiPolygon]: A unified polygon or multipolygon
        representing the area defined by the coordinates.
    """

    max_longitude = np.max(
        [coords["V0_LON"], coords["V1_LON"], coords["V2_LON"], coords["V3_LON"]]
    )
    min_longitude = np.min(
        [coords["V0_LON"], coords["V1_LON"], coords["V2_LON"], coords["V3_LON"]]
    )

    polygons = []
    if max_longitude - min_longitude > 180:
        for i in range(len(coords["V0_LON"])):
            poly1, poly2 = split_poly(
                coords["V0_LON"][i],
                coords["V0_LAT"][i],
                coords["V1_LON"][i],
                coords["V1_LAT"][i],
                coords["V2_LON"][i],
                coords["V2_LAT"][i],
                coords["V3_LON"][i],
                coords["V3_LAT"][i],
            )
            if poly1 is not None:
                min_x, _, max_x, _ = poly1.bounds
                if abs(max_x - min_x) < 10:
                    polygons.append(poly1)
            if poly2 is not None:
                min_x, _, max_x, _ = poly2.bounds
                if abs(max_x - min_x) < 10:
                    polygons.append(poly2)
    else:
        polygons = [
            Polygon(
                [
                    (coords["V0_LON"][i], coords["V0_LAT"][i]),
                    (coords["V1_LON"][i], coords["V1_LAT"][i]),
                    (coords["V2_LON"][i], coords["V2_LAT"][i]),
                    (coords["V3_LON"][i], coords["V3_LAT"][i]),
                    (coords["V0_LON"][i], coords["V0_LAT"][i]),
                ]
            )
            for i in range(len(coords["V0_LON"]))
        ]
    merged_polygon = unary_union(polygons)
    return merged_polygon  # type: ignore


class MapFlares:
    def __init__(self, datadir: str, year: str, month: str, params: dict):
        """
        Constructor of the map flares class that does monthwise processing.
        Memoize each day's shapes to avoid recomputation.

        Args:
            datadir (str): Directory containing the jsons obtained from find_flares.py.
            year (str): Year of the data.
            month (str): Month of the data.
            params (dict): Parameters.
        """
        self.datadir = datadir
        self.year = year
        self.month = month
        self.ref_elements: list[str] = params["elements"]["reference"]
        self.strict_elements: list[str] = params["elements"]["strict"]
        self.element_energies: dict[str, float] = params["elements"]["energies"]
        self.nodataval: int = params["map"]["nodataval"]
        self.height: int = params["map"]["height"]
        self.width: int = params["map"]["width"]
        self.elements = list(self.element_energies.keys())

        self.shape_memo: dict[str, Polygon | MultiPolygon] = {}
        self.transform = from_bounds(
            min_lon, min_lat, max_lon, max_lat, self.width, self.height
        )

        self.raster_weighted_ratios = {
            (ref_element, element): np.zeros(
                (self.height, self.width), dtype=np.float32
            )
            for element in self.elements
            for ref_element in self.ref_elements
        }
        self.raster_weights = {
            (ref_element, element): np.zeros(
                (self.height, self.width), dtype=np.float32
            )
            for element in self.elements
            for ref_element in self.ref_elements
        }

    def calculate_all_ratios(self) -> None:
        """
        Calculate the line weighted ratios and weights for each element with each reference element
        and stores in the class variables.

        Args:
            None

        Returns:
            None
        """

        jsons: list[str] = []
        for entry in sorted(os.listdir(f"{self.datadir}/{self.year}/{self.month}/")):
            if re.match(r"\d{4}-\d{2}-\d{2}.json", entry):
                _day = entry.split("-")[2].split(".")[0]
                jsons.append(
                    f"{self.datadir}/{self.year}/{self.month}/{self.year}-{self.month}-{_day}.json"
                )

        for json_file in tqdm.tqdm(jsons):
            _ratios, _weights = self.calculate_ratios(json_file)
            for ref_element in self.ref_elements:
                for element in self.elements:
                    self.shape_memo.clear()
                    self.raster_weighted_ratios[(ref_element, element)] += _ratios[
                        (ref_element, element)
                    ]
                    self.raster_weights[(ref_element, element)] += _weights[
                        (ref_element, element)
                    ]
                    del _ratios[(ref_element, element)]
                    del _weights[(ref_element, element)]
                    gc.collect()

    def calculate_ratios(
        self, json_file: str
    ) -> tuple[dict[tuple[str, str], npt.NDArray], dict[tuple[str, str], npt.NDArray]]:
        """
        Calculate the weighted ratios and weights for each element with each reference element for a single day.

        Args:
            json_file (str): Path to the json file.

        Returns:
            weighted_ratios (dict): Dictionary containing the weighted ratios for each element with each reference element.
            weights (dict): Dictionary containing the weights for each element with each reference element.

        """
        _raster_weighted_ratios = {
            (ref_element, element): np.zeros(
                (self.height, self.width), dtype=np.float32
            )
            for element in self.elements
            for ref_element in self.ref_elements
        }
        _raster_weights = {
            (ref_element, element): np.zeros(
                (self.height, self.width), dtype=np.float32
            )
            for element in self.elements
            for ref_element in self.ref_elements
        }

        with open(json_file, "r") as f:
            data: dict = json.load(f)
            valid_jds_elemwise = self.apply_cut(data)

            for ref_element in self.ref_elements:
                for element in self.elements:
                    valid_jds = np.intersect1d(
                        valid_jds_elemwise[ref_element],
                        valid_jds_elemwise[element],
                        assume_unique=True,
                    )

                    if valid_jds.size == 0:
                        continue

                    tiff_values = {"ratio": [], "error": []}
                    for jd in valid_jds:
                        amp, err_amp = (
                            float(x) for x in data[element][jd]["amplitude"].split("+-")
                        )
                        stddev, err_stddev = (
                            float(x) for x in data[element][jd]["stddev"].split("+-")
                        )
                        ref_amp, ref_err_amp = (
                            float(x)
                            for x in data[ref_element][jd]["amplitude"].split("+-")
                        )
                        ref_stddev, ref_err_stddev = (
                            float(x)
                            for x in data[ref_element][jd]["stddev"].split("+-")
                        )

                        ratio = (amp * stddev) / (ref_amp * ref_stddev)
                        error = ratio * np.sqrt(
                            (err_amp / amp) ** 2
                            + (err_stddev / stddev) ** 2
                            + (ref_err_amp / ref_amp) ** 2
                            + (ref_err_stddev / ref_stddev) ** 2
                        )

                        tiff_values["ratio"].append(ratio)
                        tiff_values["error"].append(error)

                    weighted_ratios, weights = self.rasterize_polygons_to_tiff(
                        list(valid_jds), tiff_values, data
                    )

                    del tiff_values["ratio"]
                    del tiff_values["error"]
                    del tiff_values
                    del valid_jds
                    gc.collect()

                    _raster_weighted_ratios[(ref_element, element)] = weighted_ratios
                    _raster_weights[(ref_element, element)] = weights

            for element in self.elements:
                del valid_jds_elemwise[element]
            del valid_jds_elemwise
            del data
            gc.collect()

        return (_raster_weighted_ratios, _raster_weights)

    def apply_cut(self, data: dict) -> dict[str, list[str]]:
        """
        Filter all the detections corresponding to each element based on the cut values.

        Args:
            data (dict): Parsed json data.

        Returns:
            valid_jds (dict): Dictionary containing the valid detection Julian dates for each element.
        """

        valid_jds: dict[str, list[str]] = {element: [] for element in self.elements}
        for element in self.elements:
            bkg_amp_std: float = data[element]["bkg_amp_std"]
            if data[element].get("threshold") is None:
                continue
            threshold: float = data[element]["threshold"]
            for jd, params in data[element].items():
                if not isinstance(params, dict):
                    continue
                if params["fit_status"] != "Success":
                    continue
                amp, _ = (float(x) for x in params["amplitude"].split("+-"))
                stddev, _ = (float(x) for x in params["stddev"].split("+-"))
                mean, _ = (float(x) for x in params["mean"].split("+-"))
                if element in self.strict_elements:
                    if (
                        amp > threshold * bkg_amp_std
                        and amp > 0
                        and np.abs(mean - self.element_energies[element]) < 0.05
                        and stddev < 0.1
                        and stddev > 0.05
                    ):
                        valid_jds[element].append(jd)
                else:
                    if (
                        amp > 0
                        and stddev > 0.05
                        and stddev < 0.15
                        and np.abs(mean - self.element_energies[element]) < 0.05
                    ):
                        valid_jds[element].append(jd)
        return valid_jds

    def rasterize_polygons_to_tiff(
        self,
        valid_jds: list[str],
        tiff_values: dict[str, list[float]],
        data: dict,
    ):
        """
        Rasterizes polygon regions and their associated tiff values into a GeoTIFF,
        using WGS84 (EPSG:4326) projection.

        Args:
            valid_jds (list[str]): List of detection IDs deemed valid by prior filtering.
            tiff_values (list[float]): Corresponding tiff values for each valid detection.
            data (dict): Dictionary containing coordinate data for each detection ID.

        Returns:
            Tuple[npt.ArrayLike, npt.ArrayLike]: Rasterized weighted ratios and weights,
            corresponding to the valid detections.
        """

        weighted_ratios = []
        weights = []

        for jd, ratio, error in zip(
            valid_jds, tiff_values["ratio"], tiff_values["error"]
        ):
            coords = data["coords"][jd]
            altitude = float(np.median(data["altitude"][jd]))
            side_of_square = 12.5 * altitude / 100
            square_coords = calculate_central_square(
                coords, side_of_square=side_of_square
            )
            try:
                polygon = self.get_polygon(jd, square_coords)
            except:
                continue
            if isinstance(polygon, MultiPolygon):
                for poly in polygon.geoms:
                    assert ratio != 0 or error != 0
                    weighted_ratios.append((poly, ratio / (error**2)))
                    weights.append((poly, 1 / (error**2)))
            else:
                assert ratio != 0 or error != 0
                weighted_ratios.append((polygon, ratio / (error**2)))
                weights.append((polygon, 1 / (error**2)))

        if len(weighted_ratios) == 0:
            return np.zeros((self.height, self.width), dtype=np.float32), np.zeros(
                (self.height, self.width), dtype=np.float32
            )
        rasterized_ratios = rasterize(
            weighted_ratios,
            out_shape=(self.height, self.width),
            transform=self.transform,
            fill=0,
            dtype=np.float32,
            merge_alg=MergeAlg.add,
        )
        rasterized_weights = rasterize(
            weights,
            out_shape=(self.height, self.width),
            transform=self.transform,
            fill=0,
            dtype=np.float32,
            merge_alg=MergeAlg.add,
        )

        del weighted_ratios
        del weights
        gc.collect()

        return rasterized_ratios, rasterized_weights

    def get_polygon(self, jd, square_coords):
        """
        Returns the polygon corresponding to the Julian date, memoizing the shape.

        Args:
            jd (str): Julian date.
            square_coords (dict): Dictionary containing the coordinates of the central square of the detection region.

        Returns:
            Polygon: Polygon corresponding to the Julian date.
        """
        if self.shape_memo.get(jd) is None:
            self.shape_memo[jd] = create_polygon(square_coords)
        return self.shape_memo[jd]


if __name__ == "__main__":
    params = parse_toml_params("config.toml")
    parser = argparse.ArgumentParser(description="Map flares to GeoTIFFs")
    parser.add_argument(
        "--datadir",
        type=str,
        help="Directory containing the data",
        default=params["map_flares"]["datadir"],
    )
    parser.add_argument(
        "--years",
        type=str,
        nargs="+",
        help="Year of the data",
        default=["2019", "2020", "2021", "2022", "2023", "2024"],
    )
    parser.add_argument(
        "--resultsdir",
        type=str,
        help="Directory to store the results",
        default=params["map_flares"]["resultsdir"],
    )
    parser.add_argument("--ncores", type=int, help="Number of cores to use", default=6)
    parser.add_argument("--rm", action="store_true", help="Remove intermediate files")
    args = parser.parse_args()

    os.makedirs(args.resultsdir, exist_ok=True)

    ref_elements = params["elements"]["reference"]
    elements = list(params["elements"]["energies"].keys())
    height = params["map"]["height"]
    width = params["map"]["width"]
    nodataval = params["map"]["nodataval"]

    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)

    def yearly_map(year):
        """
        Runs the mapping process for a single year, and saves into temporary files.

        Args:
            year (str): Year of the data.
        """
        weighted_ratios = {
            (ref_element, element): np.zeros((height, width), dtype=np.float32)
            for element in elements
            for ref_element in ref_elements
        }
        weights = {
            (ref_element, element): np.zeros((height, width), dtype=np.float32)
            for element in elements
            for ref_element in ref_elements
        }

        for month in sorted(
            filter(lambda x: x.isnumeric(), os.listdir(f"{args.datadir}/{year}/"))
        ):
            mapper = MapFlares(args.datadir, year, month, params)
            mapper.calculate_all_ratios()
            for ref_element in ref_elements:
                for element in elements:
                    weighted_ratios[(ref_element, element)] += (
                        mapper.raster_weighted_ratios[(ref_element, element)]
                    )
                    weights[(ref_element, element)] += mapper.raster_weights[
                        (ref_element, element)
                    ]
            del mapper
            gc.collect()

        for ref_element in ref_elements:
            for element in elements:
                mask = weights[(ref_element, element)] == 0
                weighted_ratios[(ref_element, element)][mask] = nodataval
                weights[(ref_element, element)][mask] = nodataval

                weighted_ratios[(ref_element, element)][~mask] = (
                    weighted_ratios[(ref_element, element)][~mask]
                    / weights[(ref_element, element)][~mask]
                )
                weights[(ref_element, element)][~mask] = 1 / np.sqrt(
                    weights[(ref_element, element)][~mask]
                )

                output_tiff = f"{args.resultsdir}/{element}_{ref_element}_{year}.tiff"
                with rasterio.open(
                    output_tiff,
                    "w",
                    driver="GTiff",
                    height=height,
                    width=width,
                    count=2,
                    dtype=np.float32,
                    crs="EPSG:4326",
                    transform=transform,
                    nodata=nodataval,
                ) as dst:
                    dst.write(weighted_ratios[(ref_element, element)], 1)
                    dst.write(weights[(ref_element, element)], 2)
                del weighted_ratios[(ref_element, element)]
                del weights[(ref_element, element)]
                gc.collect()

    pool = Pool(args.ncores)
    pool.map(yearly_map, args.years)
    pool.close()
    pool.join()

    for ref_element in ref_elements:
        for element in elements:
            combined_ratios = np.zeros((height, width), dtype=np.float32)
            combined_errors = np.zeros((height, width), dtype=np.float32)
            for year in args.years:
                output_tiff = f"{args.resultsdir}/{element}_{ref_element}_{year}.tiff"
                with rasterio.open(output_tiff, "r") as src:
                    ratios = src.read(1)
                    errors = src.read(2)
                    ratios = np.where(ratios != nodataval, ratios * (1 / errors**2), 0)
                    errors = np.where(errors != nodataval, 1 / errors**2, 0)
                    combined_ratios += ratios
                    combined_errors += errors

                if args.rm:
                    os.remove(output_tiff)

            output_tiff = f"{args.resultsdir}/{element}_{ref_element}_combined.tiff"
            mask = combined_errors == 0
            combined_ratios[mask] = nodataval
            combined_errors[mask] = nodataval
            combined_ratios[~mask] = combined_ratios[~mask] / combined_errors[~mask]
            combined_errors[~mask] = 1 / np.sqrt(combined_errors[~mask])

            with rasterio.open(
                output_tiff,
                "w",
                driver="GTiff",
                height=height,
                width=width,
                count=2,
                dtype=np.float32,
                crs="EPSG:4326",
                transform=transform,
                nodata=nodataval,
            ) as dst:
                dst.write(combined_ratios, 1)
                dst.write(combined_errors, 2)
