#!/usr/bin/python3

#   Copyright 2016-2019 Adamo Ferro
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

# 2026 port: the original version of this module used the GDAL python
# bindings (osgeo), which are not installable via pip for recent Python
# versions. GDAL has been removed completely: the module is now a
# self-contained pure-numpy ENVI reader/writer. Output files are standard
# ENVI files (data + companion .hdr, BSQ interleave, little-endian),
# readable by ENVI, GDAL and QGIS.

import os
import numpy as np

# numpy dtype name -> ENVI header "data type" code
# (https://www.l3harrisgeospatial.com/docs/ENVIHeaderFiles.html)
_NUMPY2ENVI_DATATYPE = {
    "uint8": 1,
    "int16": 2,
    "int32": 4,
    "float32": 4,
    "float64": 5,
    "complex64": 6,
    "complex128": 9,
    "uint16": 12,
    "uint32": 13,
    "int64": 14,
    "uint64": 15,
}

# ENVI header "data type" code -> numpy base dtype (byte order applied separately)
_ENVI_DATATYPE2NUMPY = {
    1: "u1",
    2: "i2",
    3: "i4",
    4: "f4",
    5: "f8",
    6: "c8",
    9: "c16",
    12: "u2",
    13: "u4",
    14: "i8",
    15: "u8",
}


def read(filename):
    hdr_filename = filename + ".hdr"
    if not os.path.isfile(hdr_filename):
        print("ERROR: cannot find ENVI header file " + hdr_filename)
        return None
    try:
        header = _read_envi_header(hdr_filename)
        samples = int(header["samples"])
        lines = int(header["lines"])
        bands = int(header.get("bands", 1))
        header_offset = int(header.get("header offset", 0))
        byte_order = int(header.get("byte order", 0))       # 0: little-endian, 1: big-endian
        endian = ">" if byte_order == 1 else "<"
        dtype = np.dtype(endian + _ENVI_DATATYPE2NUMPY[int(header["data type"])])

        # note: only the first band is read
        if bands > 1 and header.get("interleave", "bsq").lower() != "bsq":
            print("ERROR: only BSQ interleave is supported by the internal ENVI reader.")
            return None
        with open(filename, "rb") as fp:
            fp.seek(header_offset)
            np_array = np.fromfile(fp, dtype=dtype, count=samples*lines*bands)
        np_array = np_array.reshape([lines*bands, samples])
        if bands > 1:
            np_array = np_array[:lines, :]                  # keep only band 1
        return np_array
    except (IOError, OSError, KeyError, ValueError):
        print("ERROR: cannot read ENVI file " + filename)
        return None


def write(filename, np_array, complex_output=False):
    if (np_array.dtype.name == "complex64" or np_array.dtype.name == "complex128") and not complex_output:
        np_array_tmp = np.abs(np_array).astype("float64")**2.
    else:
        np_array_tmp = np_array
    try:
        return _write_numpy(filename, np_array_tmp)
    except:
        print("ERROR: problem writing image to disk.")
        return False
    return True


def _read_envi_header(hdr_filename):
    header = dict()
    with open(hdr_filename, "rt") as fp:
        line = fp.readline()
        in_braces = False       # multi-line values such as description = { ... }
        while line:
            stripped = line.strip()
            if in_braces:
                if "}" in stripped:
                    in_braces = False
            elif "=" in stripped:
                key, value = stripped.split("=", 1)
                key = key.strip().lower()
                value = value.strip()
                if value.startswith("{") and "}" not in value:
                    in_braces = True
                else:
                    header[key] = value
            line = fp.readline()
    return header


def _write_numpy(filename, np_array):
    envi_dtype = _NUMPY2ENVI_DATATYPE[np_array.dtype.name]
    hdr_filename = filename + ".hdr"
    with open(filename, "wb") as fp:
        np_array.astype(np_array.dtype.newbyteorder("<")).tofile(fp)        # little-endian (native on x86)
    with open(hdr_filename, "wt") as fp:
        fp.write("ENVI\n")
        fp.write("samples = " + str(np_array.shape[1]) + "\n")
        fp.write("lines   = " + str(np_array.shape[0]) + "\n")
        fp.write("bands   = 1\n")
        fp.write("header offset = 0\n")
        fp.write("file type = ENVI Standard\n")
        fp.write("data type = " + str(envi_dtype) + "\n")
        fp.write("interleave = bsq\n")
        fp.write("byte order = 0\n")
    return True
