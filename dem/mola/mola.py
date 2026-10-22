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

# TODO add limit checks

from dem.dem_base import DEMBase
import numpy as np
import scipy.ndimage as snd
from io_utils import envi


class MOLA(DEMBase):
    """MOLA MEGDR 1/X degrees DEM"""

    def __init__(self, dem_name="MOLA", filename_base="", image=None, dummy_value=0, max_lat=88, min_long=0, sampling_den=128):
        super().__init__(dem_name, filename_base, dummy_value)
        self._MOLA_MAX_LAT = max_lat        # degrees
        self._MOLA_MIN_LONG = min_long      # degrees
        self._MOLA_SAMPLING = 1./sampling_den       # degrees per pixel

        self._MOLA_MIN_LAT = 0              # both to be calculated when loading
        self._MOLA_MAX_LONG = 0             #

        self._mask = None

        self.deg_sampling = self._MOLA_SAMPLING
        self.offset_to_be_added = 3396000       # as per MOLA labels

        self.dem = None
        if image is not None:
            self.dem = image.copy()
            self._set_info()

    def __str__(self):
        return self.dem_name + ", base filename: " + self.filename_base + ", extension: lat " + str(self._MOLA_MIN_LAT) + ", " + str(self._MOLA_MAX_LAT) + "; long " + str(self._MOLA_MIN_LONG) + ", " + str(self._MOLA_MAX_LONG)

    def _set_info(self):
        self.dem_shape = self.dem.shape
        self._MOLA_MIN_LAT = self._MOLA_MAX_LAT - self.dem_shape[0]*self.deg_sampling
        self._MOLA_MAX_LONG = self._MOLA_MIN_LONG + self.dem_shape[1]*self.deg_sampling

    def read(self):
        if self.dem is not None:
            print("WARNING: dem has been already loaded or set by init. Not loading again.")
        elif self.filename_base != "":
            try:
                self.dem = envi.read(self.filename_base)
                self._set_info()
                return True
            except IOError:
                print("ERROR: cannot read MOLA128 DEM file.")
        else:
            print("ERROR: header filename not set.")
        return False

    def reset(self):
        del self.dem
        self.dem = None

    def get_dem_radius_from_lat_lon(self, lat, lon):
        if self.dem is not None:
            mola_x, mola_y = self.to_XY_from_LL(lat, lon)
            return self._XY_vectors_to_radius(mola_x, mola_y)
        else:
            print("ERROR: MOLA DEM not yet loaded.")
            return None

    def get_scaled_subdem_from_latlon_limits(self, lat_n, lat_s, lon_e, lon_w, scale_factor=4):
        if self.dem is not None:
            lat_n_local = np.ceil(lat_n / self.deg_sampling) * self.deg_sampling
            lat_n_local = lat_n_local if lat_n_local <= self._MOLA_MAX_LAT else self._MOLA_MAX_LAT
            lat_s_local = np.floor(lat_s / self.deg_sampling) * self.deg_sampling
            lat_s_local = lat_s_local if lat_s_local >= self._MOLA_MIN_LAT else self._MOLA_MIN_LAT

            lon_e_local = lon_e if lon_e > 0 else lon_e + 360
            lon_e_local = np.ceil(lon_e_local / self.deg_sampling) * self.deg_sampling
            lon_e_local = lon_e_local if lon_e_local <= self._MOLA_MAX_LONG else self._MOLA_MAX_LONG
            lon_w_local = lon_w if lon_w > 0 else lon_w + 360
            lon_w_local = np.floor(lon_w_local / self.deg_sampling) * self.deg_sampling
            lon_w_local = lon_w_local if lon_w_local >= self._MOLA_MIN_LONG else self._MOLA_MIN_LONG

            mola_x_min, mola_y_min = self.to_XY_from_LL(lat_n_local, lon_w_local)
            mola_x_max, mola_y_max = self.to_XY_from_LL(lat_s_local, lon_e_local)
            subdem_image = self.XY_ranges_to_radius(mola_x_min[0], mola_x_max[0], mola_y_min[0], mola_y_max[0], add_offset=False)
            scaled_subdem_image = snd.zoom(snd.gaussian_filter(subdem_image.astype("float"), 1), scale_factor, order=2) #, prefilter=False)
            # scaled_subdem_image = snd.zoom(subdem_image.astype("float"), scale_factor, order=2) #, prefilter=False)

            # # print(np.max(scaled_subdem_image-subdem_image))
            # # exit()
            # #
            # import matplotlib
            # matplotlib.use("WebAgg")
            # from matplotlib import pyplot as pp
            # pp.figure(figsize=(18, 8))
            # pp.plot(subdem_image[0, :])
            # pp.figure(figsize=(18, 8))
            # pp.plot(scaled_subdem_image[0, :])
            # pp.show()
            # exit()
            # pp.imshow(subdem_image)
            # pp.figure(figsize=(18, 8))
            # pp.imshow(scaled_subdem_image)
            # pp.show()
            # exit()

            return MOLA(image=scaled_subdem_image, max_lat=self._MOLA_MAX_LAT-mola_y_min*self.deg_sampling, min_long=self._MOLA_MIN_LONG+mola_x_min*self.deg_sampling, sampling_den=1./self.deg_sampling*scale_factor)

        else:
            print("ERROR: MOLA DEM not yet loaded.")
            return None

    def to_XY_from_LL(self, lat, long, return_as_float=False):
        mola_x = (long - self._MOLA_MIN_LONG) / self.deg_sampling
        mola_y = (self._MOLA_MAX_LAT - lat) / self.deg_sampling

        if not return_as_float:
            mola_x = np.round(mola_x).astype("int")
            mola_y = np.round(mola_y).astype("int")

        if len(mola_x.shape) == 0:
            mola_x = np.array([mola_x])
            mola_y = np.array([mola_y])

        mola_x[mola_x == self.dem_shape[1]] = 0

        self._mask = (mola_x < 0) | (mola_y < 0) | (mola_x >= self.dem_shape[1]) | (mola_y >= self.dem_shape[0])

        mola_x[mola_x < 0] = 0
        mola_y[mola_y < 0] = 0
        mola_x[mola_x >= self.dem_shape[1]] = 0
        mola_y[mola_y >= self.dem_shape[0]] = 0

        return (mola_x, mola_y)

    def to_LL_from_XY(self, x, y):
        lat = -y * self.deg_sampling + self._MOLA_MAX_LAT
        lon = x * self.deg_sampling + self._MOLA_MIN_LONG
        return (lat, lon)

    def XY_ranges_to_radius(self, x_min, x_max, y_min, y_max, add_offset=True):
        # astype(int32): numpy>=2 (NEP 50) raises OverflowError when adding the
        # MOLA radius offset (3396000 m) to an int16 DEM array (old numpy
        # silently promoted the result type)
        output_dem = self.dem[y_min:y_max+1, x_min:x_max+1].copy().astype("int32") + int(add_offset)*self.offset_to_be_added
        return output_dem

    def _XY_vectors_to_radius(self, x_vect, y_vect, add_offset=True):
        # get vector containing MOLA radius corresponding to each frame to be processed (in the same order)
        output_dem = self.dem[y_vect, x_vect].copy().astype("int32") + int(add_offset)*self.offset_to_be_added
        output_dem[self._mask] = self.dummy_value
        return output_dem


class MOLA128(MOLA):
    """MOLA MEGDR 1/128 degrees DEM"""

    def __init__(self, dem_name="MOLA128", filename_base="", image=None, dummy_value=0, max_lat=88, min_long=0):
        super().__init__(dem_name, filename_base, image, dummy_value, max_lat=max_lat, min_long=min_long, sampling_den=128)
