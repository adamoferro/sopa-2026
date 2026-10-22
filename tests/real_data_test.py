#!/usr/bin/env python3
"""
Reduced-parameter version of run_sopa_processing_2610.py, used to verify the
port against the REAL data (PDS EDR edr1260201 + MOLA128 r128 DEM) without
running the full-hours processing of the example script.
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.makedirs("/mnt/SOPA/OUTPUT/PDS", exist_ok=True)

import matplotlib
matplotlib.use("Agg")       # non-interactive

import numpy as np
from log.log import logger
from input.sharad.instrument import SHARAD
from input.sharad.data.edr.pds import EDR as EDR_PDS
from geometry.coordinate_converter import MARS_IAU2000
from dem.mola.mola import MOLA128
from processing.parameters import FocuserParameters, SimulatorParameters
from processing.simulators.sfrssim import sfRSsim
from processing.focusers.sofa import SOFA
from io_utils import envi
from plot_utils.tracks import TrackPlotOnDEM, TrackAndFirstReturnsPlotOnDEM
from plot_utils.radargrams import SimulationPlot, GroundDistancePlot, RadargramPlot

MOLA128_FILENAME_BASE = "/mnt/SOPA/DATA/dem/mola/radius/r128"
EDR_PDS_INPUT_FILENAME_BASE = "/mnt/SOPA/DATA/sharad/EDR/PDS/edr1260201/e_1260201_001_ss19_700_a"
EDR_PDS_INPUT_DATASET_NAME = "edr1260201"
EDR_PDS_OUTPUT_SIM_FILENAME_BASE = "/mnt/SOPA/OUTPUT/PDS/PDS-EDR-1260201_SIM_test"
EDR_PDS_OUTPUT_FOC_FILENAME_BASE = "/mnt/SOPA/OUTPUT/PDS/PDS-EDR-1260201_FOC_test"

# REDUCED processing load wrt the example script (verification only)
# note: the frame range must be long enough wrt METERS_SKIP, as the
# skip-distance frame selection needs at least 2 intervals
FRAME_START = 80000
FRAME_END = 81000
METERS_SKIP = 2500
FRAME_HALF_APERTURE = 64
SQUINT_ANGLE = 0

t0 = time.time()
lg = logger(verbose=True)

i = SHARAD()
cc = MARS_IAU2000()

print("reading MOLA128 DEM (2 GB)...")
m = MOLA128(filename_base=MOLA128_FILENAME_BASE)
m.read()
print("DEM read in %.1f s, shape %s, radii range: %.0f - %.0f" % (time.time()-t0, str(m.dem.shape), int(m.dem.min())+m.offset_to_be_added, int(m.dem.max())+m.offset_to_be_added))

pars = FocuserParameters(frame_start=FRAME_START, frame_end=FRAME_END, meters_skip=METERS_SKIP, frame_half_aperture=FRAME_HALF_APERTURE, squint_angle=SQUINT_ANGLE)
sim_pars = SimulatorParameters(frame_start=FRAME_START, frame_end=FRAME_END, meters_skip=METERS_SKIP, frame_half_aperture=FRAME_HALF_APERTURE, squint_angle=SQUINT_ANGLE)

s = sfRSsim(geom_obj=cc, dem_obj=m, param_obj=sim_pars, n_processes=1)

fn_base = EDR_PDS_INPUT_FILENAME_BASE
d = EDR_PDS(dataset_name=EDR_PDS_INPUT_DATASET_NAME, filename_base=fn_base, logger=lg)
t1 = time.time()
load_ok = d.load(mode="full")
print("EDR loaded in %.1f s (ok=%s): %s" % (time.time()-t1, load_ok, str(d)))
d.generate_orbit_data()
assert d.orbit_data is not None
print("orbit: %d frames, lat %.2f..%.2f, lon %.2f..%.2f" % (len(d.orbit_data["Lat"]), d.orbit_data["Lat"].min(), d.orbit_data["Lat"].max(), d.orbit_data["Lon"].min(), d.orbit_data["Lon"].max()))

tpod = TrackPlotOnDEM(m, d.orbit_data["Lat"], d.orbit_data["Lon"], title="Spacecraft ground track on DEM (full track)")
tpod.plot()

t2 = time.time()
sim_image, uncert_image, first_return_lats, first_return_lons, ground_distance_min_image, ground_distance_max_image, roi_pos = s.simulate(d.orbit_data)
print("simulation done in %.1f s: sim shape %s, sum %.1f" % (time.time()-t2, str(sim_image.shape), np.nansum(sim_image)))

envi.write(EDR_PDS_OUTPUT_SIM_FILENAME_BASE, sim_image)
envi.write(EDR_PDS_OUTPUT_SIM_FILENAME_BASE + "_uncert", uncert_image)

top_point = int(np.min(np.argmin(np.isnan(ground_distance_min_image).astype(int), axis=0))) - 50
bottom_point = top_point + 1000
sp = SimulationPlot(sim_image, top_point=top_point, bottom_point=bottom_point, title="Radargram simulation with sfRSsim\n(square facets and cos^exp function)")
sp.plot()

tfrpod = TrackAndFirstReturnsPlotOnDEM(m, d.orbit_data["Lat"], d.orbit_data["Lon"], first_return_lats, first_return_lons)
tfrpod.plot()

gdp = GroundDistancePlot(ground_distance_min_image/1000., top_point=top_point, bottom_point=bottom_point, title="Distance [km] of the closest facet contributing\nto each radargram sample simulation")
gdp.plot()

dem_radius_profile = m.get_dem_radius_from_lat_lon(d.orbit_data["Lat"], d.orbit_data["Lon"])
del m

if d.data is not None:
    f = SOFA(i, cc, dem_radius_profile, d, pars, n_processes=1, debug_mode=True)
    t3 = time.time()
    focused_rdr = f.focus()
    print("focusing done in %.1f s" % (time.time()-t3))
    if focused_rdr is not None:
        envi.write(EDR_PDS_OUTPUT_FOC_FILENAME_BASE + "_ha-" + str(pars.frame_half_aperture) + "_s-" + str(pars.squint_angle), focused_rdr)
        rp = RadargramPlot(focused_rdr, top_point=top_point, bottom_point=bottom_point, title="Radargram")
        rp.plot()
        foc_back = envi.read(EDR_PDS_OUTPUT_FOC_FILENAME_BASE + "_ha-" + str(pars.frame_half_aperture) + "_s-" + str(pars.squint_angle))
        print("focused radargram: shape %s, all finite: %s, roundtrip ok: %s" % (str(focused_rdr.shape), str(np.all(np.isfinite(focused_rdr))), str(np.allclose(foc_back, focused_rdr))))

print("TOTAL: %.1f s" % (time.time()-t0))
