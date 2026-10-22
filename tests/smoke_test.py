#!/usr/bin/env python3
"""
SOPA_2026 smoke test.

Builds a fully synthetic SHARAD-like dataset (PDS EDR header + ancillary file
+ 4-bit compressed science data + synthetic MOLA128 DEM), then runs the whole
SOPA chain on it:

    EDR load -> orbit generation -> sfRSsim simulation -> SOFA focusing
    -> ENVI/PNG output -> ENVI read-back

It is meant as a porting sanity check: it verifies that every low-level
mechanism (big-endian record parsing, bit unpacking, ENVI I/O, simulators,
focuser) works on the current Python/numpy/scipy stack. It does NOT verify
the physical correctness of the results against real reference outputs.

Run from the project root with the project virtualenv activated:

    python tests/smoke_test.py
"""

import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import matplotlib
matplotlib.use("Agg")           # non-interactive backend for the test

import numpy as np
import bitstring as bstr

from input.sharad.instrument import SHARAD
from input.sharad.data.edr.pds import EDR as EDR_PDS
from geometry.coordinate_converter import MARS_IAU2000
from dem.mola.mola import MOLA128
from processing.parameters import FocuserParameters, SimulatorParameters
from processing.simulators.sfrssim import sfRSsim
from processing.focusers.sofa import SOFA
from io_utils import envi, png as png_io
from log.log import logger


DATASET_NAME = "e_9999999"
N_FRAMES = 4000
PRI_US = 1428
PRESUMMING = 16                 # IM SS3
N_BITS_PER_BYTE = 4
N_BYTES_PER_RECORD = 1800
N_SAMPLES = 3600
DT = 3 / 80e6
DCG_DELAY = 11.98e-6
# rxwin opening time (in DT steps) chosen so that the nadir surface echo of the
# synthetic geometry (altitude ~250 km) falls inside the simulated window
RXWIN = 6674.

TMPDIR = tempfile.mkdtemp(prefix="sopa_smoke_")

PASSED = []
FAILED = []


def check(name, condition, detail=""):
    if condition:
        PASSED.append(name)
        print("  [PASS] " + name + (" (" + detail + ")" if detail else ""))
    else:
        FAILED.append(name)
        print("  [FAIL] " + name + (" (" + detail + ")" if detail else ""))


# --------------------------------------------------------------------------- #
# 1) synthetic DEM (ENVI file, same conventions as the real MOLA128 product)
# --------------------------------------------------------------------------- #
print("1) building synthetic MOLA128-like DEM...")
DEM_ROWS = 11700
DEM_COLS = 512                     # 4 degrees of longitude at 1/128 deg
rng = np.random.default_rng(42)
dem_values = (500 + 100*np.sin(np.arange(DEM_COLS)/20.) + rng.normal(0, 5, (DEM_ROWS, DEM_COLS))).astype("int16")
dem_filename_base = os.path.join(TMPDIR, "synth_dem")
envi.write(dem_filename_base, dem_values)
m = MOLA128(filename_base=dem_filename_base)
m.read()
check("DEM read (internal ENVI reader)", m.dem is not None and m.dem.shape == (DEM_ROWS, DEM_COLS), str(m.dem.shape if m.dem is not None else None))

# --------------------------------------------------------------------------- #
# 2) synthetic orbit (circular, equatorial, lon 0 -> 2 deg)
# --------------------------------------------------------------------------- #
print("2) building synthetic orbit...")
cc = MARS_IAU2000()
lat = np.linspace(-0.2, 0.2, N_FRAMES)      # slightly inclined track, like a real one
lon = np.linspace(0.1, 2.1, N_FRAMES)
sc_radius = np.full(N_FRAMES, 3396190. + 250000.) + 500.*np.sin(np.linspace(0, np.pi, N_FRAMES))       # ~250 km altitude, slowly varying
x, y, z = cc.to_xyz_from_latlon_and_radius(lat, lon, sc_radius)
vtang = np.full(N_FRAMES, 3200.)
vradial = np.zeros(N_FRAMES)

# geom record layout (copied from input/sharad/data/edr/pds.py)
geom_dt = np.dtype([("SCET_BLOCK_WHOLE", ">u4"), ("SCET_BLOCK_FRAC", ">u2"),
                ("EPHEMERIS_TIME", ">f8"), ("GEOMETRY_EPOCH", np.void, 23), ("SOLAR_LONGITUDE", ">f8"),
                ("ORBIT_NUMBER", ">i4"), ("X_MARS_SC_POSITION_VECTOR", ">f8"), ("Y_MARS_SC_POSITION_VECTOR", ">f8"),
                ("Z_MARS_SC_POSITION_VECTOR", ">f8"), ("SPACECRAFT_ALTITUDE", ">f8"), ("SUB_SC_EAST_LONGITUDE", ">f8"),
                ("SUB_SC_PLANETOCENTRIC_LATITUDE", ">f8"), ("SUB_SC_PLANETOGRAPHIC_LATITUDE", ">f8"),
                ("X_MARS_SC_VELOCITY_VECTOR", ">f8"), ("Y_MARS_SC_VELOCITY_VECTOR", ">f8"), ("Z_MARS_SC_VELOCITY_VECTOR", ">f8"),
                ("MARS_SC_RADIAL_VELOCITY", ">f8"), ("MARS_SC_TANGENTIAL_VELOCITY", ">f8"), ("LOCAL_TRUE_SOLAR_TIME", ">f8"),
                ("SOLAR_ZENITH_ANGLE", ">f8"), ("SC_PITCH_ANGLE", ">f8"), ("SC_YAW_ANGLE", ">f8"), ("SC_ROLL_ANGLE", ">f8"),
                ("MRO_SAMX_INNER_GIMBAL_ANGLE", ">f8"), ("MRO_SAMX_OUTER_GIMBAL_ANGLE", ">f8"),
                ("MRO_SAPX_INNER_GIMBAL_ANGLE", ">f8"), ("MRO_SAPX_OUTER_GIMBAL_ANGLE", ">f8"),
                ("MRO_HGA_INNER_GIMBAL_ANGLE", ">f8"), ("MRO_HGA_OUTER_GIMBAL_ANGLE", ">f8"),
                ("DES_TEMP", ">f4"), ("DES_5V", ">f4"), ("DES_12V", ">f4"), ("DES_2V5", ">f4"),
                ("RX_TEMP", ">f4"), ("TX_TEMP", ">f4"), ("TX_LEV", ">f4"), ("TX_CURR", ">f4"),
                ("CORRUPTED_DATA_FLAG", ">i2")])
geom = np.zeros(N_FRAMES, dtype=geom_dt)
geom["SCET_BLOCK_WHOLE"] = np.arange(N_FRAMES) * 100
geom["SCET_BLOCK_FRAC"] = 0
geom["EPHEMERIS_TIME"] = np.arange(N_FRAMES) * (PRI_US * PRESUMMING * 1e-6)
# note: PDS EDR ancillary files store positions in km (the reader converts to m)
geom["X_MARS_SC_POSITION_VECTOR"] = x / 1000.
geom["Y_MARS_SC_POSITION_VECTOR"] = y / 1000.
geom["Z_MARS_SC_POSITION_VECTOR"] = z / 1000.
geom["SUB_SC_EAST_LONGITUDE"] = lon
geom["SUB_SC_PLANETOCENTRIC_LATITUDE"] = lat
geom["MARS_SC_TANGENTIAL_VELOCITY"] = vtang
geom["MARS_SC_RADIAL_VELOCITY"] = vradial
geom["SOLAR_ZENITH_ANGLE"] = 45.

with open(os.path.join(TMPDIR, DATASET_NAME + "_a_a.dat"), "wb") as fp:
    fp.write(geom.tobytes())

# PDS-like header (fields must appear in the order searched by the reader)
header_txt = """PDS_VERSION_ID                 = "PDS3"
START_TIME                     = 2010-01-01T00:00:00
FILE_RECORDS                   = {n}
INSTRUMENT_MODE_ID             = SS3
MRO:PULSE_REPETITION_INTERVAL  = {pri} <MICROSECOND>
MRO:PHASE_COMPENSATION_TYPE    = "NO COMPENSATION"
MRO:COMPRESSION_SELECTION_FLAG = "STATIC"
DATA_QUALITY_ID                = "0"
END
""".format(n=N_FRAMES, pri=PRI_US)
with open(os.path.join(TMPDIR, DATASET_NAME + "_a.lbl"), "wt") as fp:
    fp.write(header_txt)

# --------------------------------------------------------------------------- #
# 3) synthetic 4-bit compressed science data (known sample values)
# --------------------------------------------------------------------------- #
print("3) building synthetic 4-bit raw echoes...")
rng2 = np.random.default_rng(7)
true_ints = rng2.integers(-8, 8, size=(N_SAMPLES, N_FRAMES)).astype("int")     # 4-bit signed values
rxwin_values = np.full(N_FRAMES, RXWIN, dtype=">f4")

with open(os.path.join(TMPDIR, DATASET_NAME + "_a_s.dat"), "wb") as fp:
    for i_f in range(N_FRAMES):
        fp.write(b"\x00" * 178)
        fp.write(rxwin_values[i_f:i_f+1].tobytes())     # slice keeps the big-endian dtype (a scalar would be converted to native order)
        fp.write(b"\x00" * 4)
        # simple, unambiguous packing: 4 bits per sample, MSB first
        ba = bstr.BitArray()
        for v in true_ints[:, i_f]:
            ba.append(bstr.BitArray(int=int(v), length=4))
        fp.write(ba.tobytes())

# --------------------------------------------------------------------------- #
# 4) EDR reading (header, ancillary, 4-bit unpacking, static decompression)
# --------------------------------------------------------------------------- #
print("4) reading synthetic EDR...")
lg = logger(verbose=False)
d = EDR_PDS(dataset_name="smoke", filename_base=os.path.join(TMPDIR, DATASET_NAME + "_a"), logger=lg)
load_ok = d.load(mode="full")
check("EDR load", load_ok is True)
check("EDR n_frames", d.n_frames == N_FRAMES, str(d.n_frames))
check("EDR pri", abs(d.pri - PRI_US*1e-6) < 1e-12, str(d.pri))
check("EDR presumming", d.presumming == PRESUMMING, str(d.presumming))

factorL = int(np.ceil(np.log2(PRESUMMING)))
factorS = factorL - N_BITS_PER_BYTE + 8
expected = true_ints * 2**factorS / PRESUMMING
data_ok = np.allclose(d.data[:100, :50], expected[:100, :50])
check("EDR 4-bit unpack + decompression", data_ok, "data shape " + str(d.data.shape))

d.generate_orbit_data()
check("orbit data generated", d.orbit_data is not None)
rxwin_expected = RXWIN*DT - DCG_DELAY + PRI_US*1e-6
check("orbit rxwin", np.allclose(d.orbit_data["rxwin_time_s"], rxwin_expected), "{:.6f}".format(d.orbit_data["rxwin_time_s"][0]))
check("orbit lat/lon", np.allclose(d.orbit_data["Lat"][::100], lat[::100]) and np.allclose(d.orbit_data["Lon"][::100], lon[::100]))

# --------------------------------------------------------------------------- #
# 5) DEM radius profile along the track
# --------------------------------------------------------------------------- #
dem_radius_profile = m.get_dem_radius_from_lat_lon(d.orbit_data["Lat"], d.orbit_data["Lon"])
check("DEM radius profile", dem_radius_profile is not None and np.all(dem_radius_profile > 3396000), "mean = {:.0f} m".format(np.mean(dem_radius_profile)))

# --------------------------------------------------------------------------- #
# 6) sfRSsim surface clutter simulation (serial)
# --------------------------------------------------------------------------- #
print("6) running sfRSsim simulation (serial, reduced parameters)...")
sim_pars = SimulatorParameters(frame_start=20, frame_end=N_FRAMES-20, meters_skip=20000, frame_half_aperture=8)
s = sfRSsim(geom_obj=cc, dem_obj=m, param_obj=sim_pars, n_processes=1)
sim_image, uncert_image, fr_lats, fr_lons, gd_min_image, gd_max_image, roi_pos = s.simulate(d.orbit_data)
check("simulation outputs", sim_image is not None and sim_image.shape[1] >= 5, "sim shape " + str(sim_image.shape))
check("simulation has energy", np.nansum(sim_image) > 0, "sum = {:.1f}".format(np.nansum(sim_image)))
check("first returns set", np.all(np.isfinite(fr_lats)) and np.all(np.isfinite(fr_lons)))

# --------------------------------------------------------------------------- #
# 7) ENVI write + read back
# --------------------------------------------------------------------------- #
print("7) testing ENVI roundtrip...")
out_base = os.path.join(TMPDIR, "sim_out")
w_ok = envi.write(out_base, sim_image)
sim_back = envi.read(out_base)
check("ENVI write", w_ok is True and os.path.isfile(out_base) and os.path.isfile(out_base + ".hdr"))
check("ENVI read-back", sim_back is not None and np.allclose(sim_back, sim_image, equal_nan=True), str(sim_back.shape if sim_back is not None else None))

# --------------------------------------------------------------------------- #
# 8) plots (non-interactive backend, only checking code paths)
# --------------------------------------------------------------------------- #
print("8) testing plot classes (Agg backend)...")
from plot_utils.tracks import TrackPlotOnDEM, TrackAndFirstReturnsPlotOnDEM
from plot_utils.radargrams import SimulationPlot, GroundDistancePlot, RadargramPlot
top_point = int(np.min(np.argmin(np.isnan(gd_min_image).astype(int), axis=0)))
plots_ok = True
try:
    TrackPlotOnDEM(m, d.orbit_data["Lat"], d.orbit_data["Lon"]).plot()
    TrackAndFirstReturnsPlotOnDEM(m, d.orbit_data["Lat"], d.orbit_data["Lon"], fr_lats, fr_lons).plot()
    SimulationPlot(sim_image, top_point=top_point, bottom_point=top_point+500).plot()
    GroundDistancePlot(gd_min_image/1000., top_point=top_point, bottom_point=top_point+500).plot()
except Exception as ex:
    plots_ok = False
    print("      plot exception:", repr(ex))
check("plot classes run", plots_ok)

# --------------------------------------------------------------------------- #
# 9) SOFA focusing (serial)
# --------------------------------------------------------------------------- #
print("9) running SOFA focuser (serial, reduced parameters)...")
i = SHARAD()
foc_pars = FocuserParameters(frame_start=20, frame_end=N_FRAMES-20, meters_skip=20000, frame_half_aperture=8, squint_angle=0)
f = SOFA(i, cc, dem_radius_profile, d, foc_pars, n_processes=1, debug_mode=False)
focused_rdr = f.focus()
check("focusing output", focused_rdr is not None and focused_rdr.shape[1] >= 5, "foc shape " + str(focused_rdr.shape))
check("focusing has finite energy", focused_rdr is not None and np.all(np.isfinite(focused_rdr)) and np.sum(focused_rdr) > 0)

foc_base = os.path.join(TMPDIR, "foc_out")
envi.write(foc_base, focused_rdr)
focused_back = envi.read(foc_base)
check("focused ENVI roundtrip", focused_back is not None and np.allclose(focused_back, focused_rdr))

# PNG writer
png_ok = png_io.write(os.path.join(TMPDIR, "foc_out.png"), focused_back)
check("PNG write", png_ok is True and os.path.isfile(os.path.join(TMPDIR, "foc_out.png")))
try:
    RadargramPlot(focused_rdr, top_point=top_point, bottom_point=top_point+500).plot()
    radargram_plot_ok = True
except Exception as ex:
    radargram_plot_ok = False
    print("      radargram plot exception:", repr(ex))
check("RadargramPlot runs", radargram_plot_ok)

# --------------------------------------------------------------------------- #
shutil.rmtree(TMPDIR)
print()
print("=" * 60)
print("SMOKE TEST: %d passed, %d failed" % (len(PASSED), len(FAILED)))
if FAILED:
    print("FAILED checks:", ", ".join(FAILED))
    sys.exit(1)
print("ALL CHECKS PASSED")
