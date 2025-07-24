# README


This folder contains:
- `*.py` -- Files necessary to run the detection and mapping of XRF lines using the CLASS L1 data files
- `clustering_notebooks/*.ipynb` -- Jupyter Notebooks for clustering and analyzing element ratio groups (`clustering_notebooks/*.tiff`)
- `config.toml` -- The TOML file used for basic configs by all code files
- `environment.txt` -- File to create conda environment
- This `README.md`

## Installation
The code is written in Python, and the corresponding dependencies can be installed as a [conda](https://www.anaconda.com/) environment using the provided `environment.txt` file. The environment can be created using the following command:

```bash
conda create --name isro --file environment.txt
```

Miniconda, a lightweight distribution of Anaconda, can be installed following the instructions [here](https://docs.anaconda.com/miniconda/#quick-command-line-install), and the environment can be activated using the following command:

```bash
conda activate isro
```

**Now, we need to install the packages in the conda env using pip. We install using pip as conda was not able to resolve dependencies for all the required packages, therefore we include a `requirements.txt` as well.**

```bash
pip install -r requirements.txt
```

All the python files presented below have a `--help` argument that can be used to display the help message and the arguments that can be passed to the code. 

### config.toml

The following `config.toml` file provided below is used by all the programs to supply basic settings. It must be present in the same directory as the code files. 

**All the codes can be run without passing the parameters which are already present in config. The config parameters are taken as default arguments to the parser**

```
[map]
width = 2048
height = 1024
nodataval = 9999

[elements]
reference = ["Al", "Si", "Mg", "O"]
strict = ["Al", "Si", "Mg"]

[elements.energies]
# FeL should be after O, for its succesful fitting, a explained in our report 
"O" = 0.525
"Mg" = 1.25
"Al" = 1.48
"Si" = 1.74
"CaKa" = 3.69
"CaKb" = 4.01
"Ti" = 4.51
"Cr" = 5.41
"Mn" = 5.90
"FeK" = 6.40
"FeL" = 0.71

[find_flares]
# The following parameters are used in find_flares.py
timebin = 96             # seconds: Time bin for detection
datadir = "./data"       # Directory containing the data
resultsdir = "./results" # Directory to save the results
Nsigma = true            # Whether to use Nsigma detection threshold
fit_func = "curve_fit"   # Fitting function to use

[map_flares]
# The following parameters are used in map_flares.py
datadir = "./results" # Directory containing the data
resultsdir = "./maps" # Directory to save the results

[gen_catalog]
# The following parameters are used in gen_catalog.py
datadir = "./results" # Directory containing the data
resultsdir = "./maps" # Directory to save the results

[split_data_and_process_bkg]
# The following parameters are used in split_data_and_process_bkg.py
datadir = "./data"       # Directory containing the data
```

## Splitting Data and Background Processing

`split_data_and_process_bkg.py` contains Python code to split the data files into the $\texttt{light}$ and $\texttt{dark}$ directories based on the $\texttt{SOLARANG}$ keyword in the fits headers. It expects a directory structure analogous to the one described below, which is similar to the default structure of the CLASS L1 data files. We use multiprocessing to speed up the process of sorting the data files. 

```
.
└── year
    └── month
        └── day
            *.fits
```

After splitting the data files into $\texttt{light}$ and $\texttt{dark}$ directories, the program also checks each $\texttt{dark}$ FITS file if is actually occulted $(R(\csc (\texttt{SOLARANG}) - 1) > \texttt{SATALT})$. Each occulted data is classified by phase of the moon, (New Moon, First Quarter, Third Quarter, Full Moon) each spanning $\approx 7$ days. For each occulted 8 s spectra of a given year the $\texttt{COUNTS}$ and $\texttt{MIDUTC}$ are appended into a masterfile `data/year/phase_masterfile.fits` based on phase. Finally, all yearwise masterfiles are combined into a main masterfile in the `data/phase_masterfile.fits`

__Note:__ The code will create the `light` and `dark` directories in the same directory as the data files and move the corresponding files to the respective directories. The masterfiles are created in corresponding locations and overwritten on subsequent runs.

### Dynamic Updates

If new data needs to be added dynamically, say for a new year, create a new directory for the data and run the file for this year. This will process only the new year and update the masterfiles. The background model in flare detection is made by averaging the previous 12 months from the queried date.

Example command:
```bash
python split_data.py --datadir "/path/to/data" --ncores 4 --year 2020
```

#### Arguments
- `datadir` -- data directory
- `ncores` -- number of cores, default 8
- `year` -- year to run process, default `None` which processes all years


## Solar Flare and Potential Line Detections

`find_flares.py` has Python code to detect flares given the data directory, timebin, results directory and a specific year, month, day or a `--runall` argument which will run the code for all the days in the data directory. The results directory is automatically created if it does not exist and contains the following files:
- `year-month-day.json`: JSON file containing the amplitudes, gaussian fit params, and Julian dates of potential flares for the specific day.
- `year-month-day.png`: PNG file containing the plot of the lightcurve for the day.
- `year-month-day.mp4`: MP4 file containing the animation of the $\texttt{light}$ spectra for the day. This is only produced if the `--animate` argument is supplied.
    
The code expects the following directory structure for the data:
```
.
├── year
│   └── month
│       └── day
│           ├── dark (not required)
│           │       *.fits
│           └── light
│                   *.fits
├── new_moon_masterfile.fits
├── first_quarter_masterfile.fits
├── third_quarter_masterfile.fits
└── full_moon_masterfile.fits
```
There is an optional argument `--animate` which will animate the spectra in the form of a `.mp4` file.

_Note: The animate function requires `ffmpeg` to be installed on the system. It can be installed on Debian based systems using the following command:_
```bash
sudo apt install ffmpeg
```
_For other systems, please refer to the [ffmpeg website](https://ffmpeg.org/download.html) for installation instructions._



Example command:
```bash
python find_flares.py --datadir "/path/to/data" --timebin 96 --resultsdir "/path/to/results" --year "2024" --month "01" --day "01" --animate
```
    
#### Arguments
- `--timebin` -- Time bin in seconds for binning the light and dark data (default taken from config)
- `--datadir` -- Path to the directory containing the data files and should be supplied as a string (default taken from config)
- `--resultsdir` -- Path to the directory where the results will be saved and should be supplied as a string (default taken from config)
- `--year`, `--month` and `--day` are the year, month and day for which the code will be run, respectively and must also be provided as a string
-  `--animate` -- Optional and stored true if supplied

We can run the code for a particular year or month just by passing the `--runall` argument along with the `--year` or `--month` argument.
Example command to run for all days in the year 2024:
```bash
python find_flares.py --datadir "/path/to/data" --timebin 96 --resultsdir "/path/to/results" --year "2024" --runall  --animate
```

This will produce `year-month-day.json`, `year-month-day.png`, `year-month-day_bkg.png`, `year-month-day.mp4` files for each day in 2024, sorted by month in the resultsdir.

## Catalog Flares

`gen_catalog.py` contains Python code to create a catalog of confident line detections after applying constraints on the fitted gaussian for each element. The catalog will be created in `resultsdir/catalog.json` in the following JSON format :- `{jd : {element : {flux : float64 , err_flux: float64}}}`, containing JD of valid detection along with element flux and error in flux reported from the fitting model.

The program expects the following directory structure in `data_dir`:
```
.
└── year
   └── month
           year_month_day.json
```


Example command:
```bash
python gen_catalog.py --datadir "/path/to/data" --resultsdir "/path/to/results" --years "2024"
```

#### Arguments

- `--data_dir` -- Input directory containing the potential flares generated by `find_flares.py` (default taken from config)
- `--resultsdir` -- Output directory to place the catalogs in (default taken from config)
- `--years` -- Space-separated list of years to process catalogs for, default is all years


## Mapping Ratios
 
`map_flares.py` contains Python code to map flux ratios of flares detected by the `find_flares.py` code given the `--datadir` as input with the following structure (this structure is automatically produced by the `find_flares.py` code).
```
    .
    └── year
        └── month
                year-month-day.json
```

```
"jd", "element", "ratio", "longitude", "latitude"
```

The results directory is automatically created if it doesnt exist which can be set using the `--resultdir` argument and contains all the generated maps. The maps are generated separately for each element, and are in the form of a GeoTIFF with Coordinate Reference System (CRS) set as `EPSG:4326`.

The Jupyter Notebook `maps/analyze_tiff.ipynb` contains interactive code to visualize any of the generated TIFF files including ratios and error maps.



Example Command:
```bash
python map_flares.py --datadir "/path/to/data" --resultsdir "/path/to/results" --years 2019 2020 --rm --ncores 7
```
This will generate the maps and the catalog for each year in the data directory and save them in the results directory. 

#### Arguments

- `--datadir` -- Input directory path (default taken from config)
- `--resultsdir` -- Output directory path (default taken from config)
- `--years` -- Space separated list of years, default is all years
- `--ncores` -- Number of cores to use, default 6
- `--rm` -- Optional argument which will clear intermediate files created while processing

## Dynamic and Interactive Visualization Tool


- Frontend Framework: Built using React (JavaScript Library) to create a dynamic and user-friendly interface.
- TIFF File Conversion: Python script leverage GDAL to convert TIFF files into tiles, enabling efficient rendering.

```
./webpage
├── public/                 # Static files served by React
├── src/                    # Source files for the React app
│   ├── components/         # Reusable React components
│   └── index.js               # Entry point of the React app
└── package.json               # Node.js dependencies and project metadata
```

To view the ratio and cluster maps on the interactive visualization browser tool, run the following commands in this directory:

```bash
cd ./webpage
npm i
npm start
```

Open any browser of your choice and open `http://localhost:3000`. Refer to the video for a demo of using the tool.


_Note: The interactive browser tool requires `node` and `npm` to be installed on the system. They can be installed on Debian based systems using the following command:_
```bash
sudo apt install nodejs
sudo apt install npm
```
_For other systems, please refer to the [npm website](https://docs.npmjs.com/downloading-and-installing-node-js-and-npm) for installation instructions._


## Element ratio group clustering

The `clustering_notebooks/` directory contains a set of Jupyter notebooks containing clustering methods, clusters, and lunar base map visualization of clusters for the best observed elemental ratios.

- `Al_by_Si_vs_Mg_by_Si.ipynb` -- Clustering and visualization for Mg/Si and Al/Si.
- `Fe_by_Si_vs_Al_by_Si.ipynb` -- Clustering and visualization for Fe/Si and Al/Si.
- `Mg_num_vs_Al_by_Si.ipynb` -- Clustering and visualization for $\text{Mg}$ number and Al/Si.

Each notebook contains documentation of how to run the notebook and generate all plots, along with observations and inferences drawn (generated outputs are also present).
