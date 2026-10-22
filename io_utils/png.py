#!/usr/bin/python3

#   Copyright 2016-2022 Adamo Ferro
#
#   This file is part of SOPA.
#
#   SOPA is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
#
#   SOPA is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#   GNU General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with SOPA. If not, see <http://www.gnu.org/licenses/>.
#
#   The use of SOPA or part of it for the creation of any sub-product
#   (e.g., scientific papers, posters, images, other softwares)
#   must be acknowledged.

import sys
import png
import numpy as np

TOP_NOISE_SAMPLES = 100

# in case of real data
DYN_RANGE_LOW_DB = -3
DYN_RANGE_HIGH_DB = 30

# in case of simulated data (no noise)
PERC_MIN_THR = 70
PERC_MAX_THR = 99.8

def write(filename, np_array, sim_data=False):
    if (np_array.dtype.name == "complex64" or np_array.dtype.name == "complex128"):
        np_array_tmp = np.abs(np_array).astype("float64")**2.
    else:
        np_array_tmp = np_array

    # noise mean estimation (it's complicated because PDS RDRs start with zeros)
    mask = np_array_tmp!=0
    first_nonzero_values = np.where(mask.any(axis=0), mask.argmax(axis=0), -1)
    cols_not_ok = np.where(first_nonzero_values == -1)[0]
    if len(cols_not_ok) == np_array_tmp.shape[1]:       # if all cols are not ok
        print("ERROR: cannot determine noise mean")
        return False

    # mask values not considered for noise mean calculation with nans
    noise_array_tmp = np_array_tmp.copy()
    first_nonzero_values[first_nonzero_values == -1] = 0        # reset to zero, but finally not considered (see next lines)
    mask = np.ones(noise_array_tmp.shape) * np.nan
    for i_col in np.arange(noise_array_tmp.shape[1]):
        mask[:, first_nonzero_values[i_col]:first_nonzero_values[i_col]+TOP_NOISE_SAMPLES] = 1
    if len(cols_not_ok) > 0:
        noise_array_tmp[:, cols_not_ok] = np.nan

    noise_mean = np.nanmean(noise_array_tmp)

    if not sim_data:
        # real data, dB conversion
        noise_mean_db = 10.*np.log10(noise_mean)
        np_array_tmp[np_array_tmp == 0] = sys.float_info.min
        np_array_tmp = 10.*np.log10(np_array_tmp)
        LOW_THR = noise_mean_db-DYN_RANGE_LOW_DB
        HIGH_THR = noise_mean_db+DYN_RANGE_HIGH_DB
    else:
        LOW_THR = np.percentile(np_array_tmp, PERC_MIN_THR)
        HIGH_THR = np.percentile(np_array_tmp, PERC_MAX_THR)

    # thresholding and stretching
    np_array_tmp[np_array_tmp<LOW_THR] = LOW_THR
    np_array_tmp[np_array_tmp>HIGH_THR] = HIGH_THR
    np_array_tmp = ((np_array_tmp - LOW_THR) / (HIGH_THR - LOW_THR) * 255).astype("uint8")

    try:
        png.from_array(np_array_tmp.copy(), 'L').save(filename)     # .copy() needed to avoid a pypng "bug" with some numpy ndarrays
    except:
        print("ERROR: problem writing image to disk.")
        return False
    return True
