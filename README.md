# SOPA 2026 - SHARAD Open Processing Attempt (2026 port)

This is a port of the original SOPA suite (see `SOPA_OLD/`, README below) to a
current Linux + Python 3 stack. The code structure and the algorithms are
**unchanged**: only the minimum modifications needed to run on modern
libraries have been applied (see "What changed" below).

SOPA is a suite of open source algorithms for SHAllow RADar (SHARAD) low-level
data processing. At the time of writing the suite includes:
- data reading with common interface for:
  - PDS Italian EDR raw data (echoes and geometric data);
  - PDS Italian RDR focused data (echoes and geometric data);
  - CO-SHARPS raw data (echoes and full ancillary dataset);
  - PDS US RDR focused data (echoes and geometric data);
- an "essential" version of SOFA (SHARAD Open Focusing Attempt), a focuser
  that, besides standard nadir focusing, allows squinted synthetic aperture
  radar processing;
- surface clutter simulators:
  - dRSsim, a dummy radar sounder surface clutter simulator;
  - sfRSsim, an enhanced simulator using square facets that can use a cos^exp
    law to weight the effect of surfaces not facing the radar;
- plotting tools for quick visualization.

The main example program is `run_sopa_processing_2610.py`.

## Requirements

Tested on Ubuntu 26.04, Python 3.14 with:
- numpy 2.5.4
- scipy 1.18.1
- matplotlib 3.11.2
- PyQt6 6.11 (GUI backend, needed to actually show the figures; without it
  matplotlib silently falls back to the non-interactive Agg backend and
  `plt.show()` does nothing)
- pandas 3.0.6 (only needed by `processing/simulators/sfrssim_alt_angle.py`)
- bitstring 5.0.0 (only needed for PDS Italian EDR data)
- pypng 0.20220715.0 (only needed for `io_utils/png.py`)

GDAL is **not** used anymore: `io_utils/envi.py` is a self-contained
pure-numpy ENVI reader/writer (data + companion `.hdr`, BSQ interleave,
little-endian). Output files are standard ENVI files readable by ENVI, GDAL
and QGIS.

Everything is installed in a **virtual environment**; no system-wide Python
packages are required.

## Setup

The virtual environment used on this machine is `/home/ferroa/venvs/sopa_2026`
(it lives outside `/mnt/SOPA` because VirtualBox shared folders do not support
symlinks, which venvs need). To recreate it from scratch:

```bash
python3 -m venv /home/ferroa/venvs/sopa_2026        # add --without-pip + get-pip.py if python3.14-venv is not installed
source /home/ferroa/venvs/sopa_2026/bin/activate
pip install -r requirements.txt
```

## Running

```bash
source /home/ferroa/venvs/sopa_2026/bin/activate
cd /mnt/SOPA/SOPA_2026
python run_sopa_processing_2610.py
```

The example processes ~20000 frames of EDR orbit 1260201 (simulation + full
squint-0 focusing): it takes a long time (hours) on a single core, since the
ray-based parallelization has been removed (see below). For a quick check,
run the reduced-parameter version instead:

```bash
python tests/real_data_test.py      # real data, reduced frame range (~1-2 min)
python tests/smoke_test.py          # fully synthetic data, no real data needed
```

Plots are interactive (matplotlib Qt backend via PyQt6): each `plot()` call
opens a window and blocks until it is closed, as in the original SOPA. When
running headless (no display), set `MPLBACKEND=Agg` in the environment; the
figures will then not be shown (the test scripts do this themselves).

## Parallelization status

The original code used the `ray` library (v0.7.3) for parallel simulation and
focusing. Ray has been **removed**: `n_processes` arguments are still accepted
for interface compatibility but ignored (values > 1 fall back to 1 with a
notice). A new parallelization approach is to be decided; candidates include
`concurrent.futures.ProcessPoolExecutor` (with fork-based copy-on-write
sharing of the large read-only DEM arrays on Linux), `joblib`, or `dask`.

## What changed w.r.t. SOPA_OLD

Minimum-change port, no restructuring:

- `processing/simulators/sim_base.py`, `processing/focusers/sofa.py`: removed
  the `ray` import and the parallel branches; the original serial code paths
  are preserved and always used.
- `input/sharad/data/edr/pds.py`, `input/sharad/data/rdr/pds.py`: `np.fromstring`
  (removed from numpy) replaced by `np.frombuffer`.
- `processing/focusers/sofa.py`: `scipy.ndimage.filters` module (removed from
  scipy) replaced by `scipy.ndimage`.
- `plot_utils/radargrams.py`, `plot_utils/tracks.py`: `matplotlib.cm.get_cmap`
  and `plt.cm.<name>` attribute access (removed/deprecated) replaced by
  `matplotlib.colormaps[...]`.
- `processing/simulators/sfrssim_alt_angle.py`: `DataFrame.append` (removed in
  pandas 2.0) replaced by `pd.concat`.
- `geometry/coordinate_converter.py`: `np.arange` stop value converted to
  scalar (`distances_cum[-1, 0]`), numpy>=2 does not convert size-1 arrays to
  scalars anymore.
- `dem/mola/mola.py`: DEM radius arrays are upcast to `int32` before adding the
  MOLA offset (numpy>=2 / NEP 50 raises OverflowError when adding 3396000 to an
  `int16` array; old numpy silently promoted the result).
- `io_utils/envi.py`: GDAL removed completely; replaced by a self-contained
  pure-numpy ENVI reader/writer (see Requirements).
- `run_sopa_processing_2610.py`: data paths updated to this machine
  (`/mnt/SOPA/DATA/...`, `/mnt/SOPA/OUTPUT/PDS/...`), the `simulate()`
  return unpack updated (the function returns one extra value, `roi_pos`,
  added after the original example was written), and `MOLA128` now imported
  from `dem.mola.mola` instead of the (historically separate) `dem.mola.mola128`
  module, which is missing from the old snapshot.
- `tests/smoke_test.py`, `tests/real_data_test.py`: new test scripts (see
  Running).

---

# SOPA - SHARAD Open Processing Attempt (original README)

SOPA is thought to be a suite of open source algorithms for SHAllow RADar (SHARAD) low-level data processing written in python v3.

At the time of writing the suite includes:
- data reading with common interface for:
  - PDS Italian EDR raw data (echoes and geometric data*);
  - PDS Italian RDR focused data (echoes and geometric data*)
  - CO-SHARPS raw data (echoes and full ancillary dataset);
  - PDS US RDR focused data (echoes and geometric data**);
- an "essential" version of SOFA (SHARAD Open Focusing Attempt, http://af-projects.it/sofa), a focuser that, besides standard nadir focusing, allows squinted synthetic aperture radar processing***;
- surface clutter simulators****:
  - dRSsim, a dummy radar sounder surface clutter simulator;
  - sfRSsim, an enhanced simulator using square facets that can use a cos^exp law to weight the effect of surfaces not facing the radar;
- plotting tools for quick visualization of focused radargrams, simulations, ground distance images, and first return maps superimposed to the DEM.

Unfortunately, almost no documentation is provided. An example program showing how to use the main functions can be found in the root of the project.

*Orbits extracted from PDS Italian data may be not sufficiently precise for data acquired before 2013.

**Orbits extracted from PDS US RDR data may be not sufficiently precise for supporting simulations and or focusing.

***By now, SOFA can handle only night time radargrams as no correction of ionosphere effects has been implemented yet.

****The simulation principles are similar to those described in http://dx.doi.org/10.1109/TGRS.2012.2219315.


Feel free to contact me for any help (contacts here: http://af-projects.it/contacts).

***NOTE:** This project has not been developed during my work at the University of Trento, nor it has been supported by the University of Trento in any way.*

### Requirements (original)
SOPA has been tested only in a Linux environment using python v3.5 and the following modules:
- sys
- math
- os
- glob (needed only for parsing CO-SHARPS data)
- datetime
- gzip (needed only if txt files related to CO-SHARPS ancillary data have been downloaded gzipped)
- numpy 1.17.4
- scipy 1.3.2
- bitstring (needed only for PDS Italian EDR data)
- ray 0.7.3 (needed only for parallel computation in surface clutter simulators and SOFA, not mandatory)
- matplotlib 3.0.3 (needed only for plotting classes)

### Current version
0.02

### Disclaimer

The author does not guarantee that this software will always provide correct results nor that it will not crash your hardware. In any case, any use of SOPA is ONLY user responsibility. The use of SOPA or part of it for the creation of any sub-product (e.g., scientific papers, posters, images, other softwares) must be acknowledged.
As you may imagine, the code is not optimized for fast running.
