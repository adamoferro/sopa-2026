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

from matplotlib import pyplot as pp
import matplotlib.patches as mpatches
import numpy as np


class TrackPlotOnDEM(object):
    """docstring"""

    def __init__(self, dem_obj, track_lats, track_lons, auto_show=False, title=None):
        self._dem_obj = dem_obj
        self._track_lats = track_lats
        self._track_lons = track_lons
        self._n_labels = 10
        self._auto_show = auto_show
        if title is not None:
            self._title = title
        else:
            self._title = "Spacecraft ground track on DEM [km]"

    def _calculate_xy_limits(self, lats, lons):
        sc_track_x, sc_track_y = self._dem_obj.to_XY_from_LL(lats, lons, return_as_float=True)
        min_x = int(np.floor(np.min(sc_track_x)))
        max_x = int(np.ceil(np.max(sc_track_x)))
        min_y = int(np.floor(np.min(sc_track_y)))
        max_y = int(np.ceil(np.max(sc_track_y)))
        return min_x, max_x, min_y, max_y, sc_track_x, sc_track_y

    def _create_labels(self, max_lat, min_lat, max_long, min_long):
        lat_range = np.round(np.arange(max_lat, min_lat, -self._dem_obj.deg_sampling)*100)/100
        lat_range_len = len(lat_range)
        long_range = np.round(np.arange(min_long, max_long, self._dem_obj.deg_sampling)*100)/100
        long_range_len = len(long_range)
        step_x = int(long_range_len / (self._n_labels - 1))
        step_y = int(lat_range_len / (self._n_labels - 1))
        labels_x_pos = np.arange(0, long_range_len, step_x)
        labels_y_pos = np.arange(0, lat_range_len, step_y)
        labels_x = long_range[::step_x]
        labels_y = lat_range[::step_y]
        return labels_x, labels_y, labels_x_pos, labels_y_pos

    def plot(self):
        min_x, max_x, min_y, max_y, sc_track_x, sc_track_y = self._calculate_xy_limits(self._track_lats, self._track_lons)

        min_lat, min_long = self._dem_obj.to_LL_from_XY(min_x, max_y)
        max_lat, max_long = self._dem_obj.to_LL_from_XY(max_x, min_y)

        labels_x, labels_y, labels_x_pos, labels_y_pos = self._create_labels(max_lat, min_lat, max_long, min_long)

        pp.figure(figsize=(12, 6))
        pp.imshow(self._dem_obj.XY_ranges_to_radius(min_x, max_x+1, min_y, max_y+1)/1000, interpolation="none")
        pp.plot(sc_track_x-min_x, sc_track_y-min_y, "r")
        pp.axis("tight")
        pp.title(self._title, fontweight="bold")
        pp.xlabel("Long", fontweight="bold")
        pp.xticks(labels_x_pos, labels_x)
        pp.ylabel("Lat", fontweight="bold")
        pp.yticks(labels_y_pos, labels_y)
        pp.colorbar()
        self.show(explicit=False)

    def show(self, explicit=True):
        if explicit or self._auto_show:
            pp.show()


class TrackAndFirstReturnsPlotOnDEM(TrackPlotOnDEM):
    """docstring"""

    def __init__(self, dem_obj, track_lats, track_lons, fr_lats, fr_lons, auto_show=False, title=None):
        super().__init__(dem_obj, track_lats, track_lons, auto_show=auto_show, title=None)
        self._fr_lats = fr_lats
        self._fr_lons = fr_lons
        if title is not None:
            self._title = title
        else:
            self._title = "Spacecraft ground track (yellow) vs.\nsimulated first return positions (magenta) on DEM [km]"

    def plot(self):
        min_x, max_x, min_y, max_y, fr_x, fr_y = self._calculate_xy_limits(self._fr_lats, self._fr_lons)
        _, _, _, _, sc_track_x, sc_track_y = self._calculate_xy_limits(self._track_lats, self._track_lons)

        sc_track_x_ok_ids = (sc_track_x >= min_x) & (sc_track_x <= max_x)
        sc_track_y_ok_ids = (sc_track_y >= min_y) & (sc_track_y <= max_y)
        sc_track_ok_ids = sc_track_x_ok_ids & sc_track_y_ok_ids
        sc_track_x = sc_track_x[sc_track_ok_ids]
        sc_track_y = sc_track_y[sc_track_ok_ids]

        min_lat, min_long = self._dem_obj.to_LL_from_XY(min_x, max_y)
        max_lat, max_long = self._dem_obj.to_LL_from_XY(max_x, min_y)

        labels_x, labels_y, labels_x_pos, labels_y_pos = self._create_labels(max_lat, min_lat, max_long, min_long)

        pp.figure(figsize=(12, 6))
        pp.imshow(self._dem_obj.XY_ranges_to_radius(min_x, max_x+1, min_y, max_y+1)/1000, interpolation="none")
        pp.plot(sc_track_x-min_x, sc_track_y-min_y, "y")
        pp.plot(fr_x-min_x, fr_y-min_y, "+", color="magenta", markersize=3)
        pp.axis("tight")
        pp.title(self._title, fontweight="bold")
        pp.xlabel("Long", fontweight="bold")
        pp.xticks(labels_x_pos, labels_x)
        pp.ylabel("Lat", fontweight="bold")
        pp.yticks(labels_y_pos, labels_y)
        pp.colorbar()
        self.show(explicit=False)


class TrackAndROIReturnsPlotOnDEM(TrackPlotOnDEM):
    """docstring"""

    standard_cmap = np.array([
                     [0.12156862745098039, 0.4666666666666667, 0.7058823529411765],
                     [1.0, 0.4980392156862745, 0.054901960784313725],
                     [0.17254901960784313, 0.6274509803921569, 0.17254901960784313],
                     [0.8392156862745098, 0.15294117647058825, 0.1568627450980392],
                     [0.5803921568627451, 0.403921568627451, 0.7411764705882353],
                     [0.5490196078431373, 0.33725490196078434, 0.29411764705882354],
                     [0.8901960784313725, 0.4666666666666667, 0.7607843137254902],
                     [0.4980392156862745, 0.4980392156862745, 0.4980392156862745],
                     [0.7372549019607844, 0.7411764705882353, 0.13333333333333333],
                     [0.09019607843137255, 0.7450980392156863, 0.8117647058823529]
                    ])
    standard_cmap_n_colors = len(standard_cmap)

    ref_max_angle = np.deg2rad(1)
    ref_max_alt_angle_deg = 1

    def __init__(self, dem_obj, track_lats, track_lons, roi_pos, plot_alt_angle=False, auto_show=False, title=None):
        super().__init__(dem_obj, track_lats, track_lons, auto_show=auto_show, title=None)
        self._roi_lats = roi_pos[0]
        if len(self._roi_lats) > 0:
            self._NO_ROIS = False
            self._roi_lons = roi_pos[1]
            self._roi_ias = None
            self._roi_alt_ias = None
            self._plot_alt_angle = plot_alt_angle
            if len(roi_pos) == 4:
                self._roi_ias = roi_pos[2]
                self._roi_alt_ias = roi_pos[3]
            if title is not None:
                self._title = title
            else:
                self._title = "Spacecraft ground track (yellow) vs.\nsimulated positions of ROI returns on DEM [km]"
        else:
            self._NO_ROIS = True

    def plot(self):
        if self._NO_ROIS:
            print("ERROR: no ROIs to plot.")
        else:
            roi_x = dict()
            roi_y = dict()
            tmp_min_x = list()
            tmp_max_x = list()
            tmp_min_y = list()
            tmp_max_y = list()
            for count, ir in enumerate(self._roi_lats):
                if len(self._roi_lats[ir]) > 0:
                    tmx, tMx, tmy, tMy, roi_x[ir], roi_y[ir] = self._calculate_xy_limits(self._roi_lats[ir], self._roi_lons[ir])
                    tmp_min_x.append(tmx)
                    tmp_max_x.append(tMx)
                    tmp_min_y.append(tmy)
                    tmp_max_y.append(tMy)
            min_x = int(min(tmp_min_x))
            max_x = int(max(tmp_max_x))
            min_y = int(min(tmp_min_y))
            max_y = int(max(tmp_max_y))

            _, _, _, _, sc_track_x, sc_track_y = self._calculate_xy_limits(self._track_lats, self._track_lons)

            sc_track_x_ok_ids = (sc_track_x >= min_x) & (sc_track_x <= max_x)
            sc_track_y_ok_ids = (sc_track_y >= min_y) & (sc_track_y <= max_y)
            sc_track_ok_ids = sc_track_x_ok_ids & sc_track_y_ok_ids
            sc_track_x = sc_track_x[sc_track_ok_ids]
            sc_track_y = sc_track_y[sc_track_ok_ids]

            min_lat, min_long = self._dem_obj.to_LL_from_XY(min_x, max_y)
            max_lat, max_long = self._dem_obj.to_LL_from_XY(max_x, min_y)

            labels_x, labels_y, labels_x_pos, labels_y_pos = self._create_labels(max_lat, min_lat, max_long, min_long)

            pp.figure(figsize=(12, 6))
            pp.imshow(self._dem_obj.XY_ranges_to_radius(min_x, max_x+1, min_y, max_y+1)/1000, interpolation="none")
            pp.colorbar()
            pp.plot(sc_track_x-min_x, sc_track_y-min_y, "y*")

            if (not self._plot_alt_angle and self._roi_ias is None) or (self._plot_alt_angle and self._roi_alt_ias is None):
                for ir in roi_x:
                    pp.scatter(x=roi_x[ir]-min_x, y=roi_y[ir]-min_y, s=0.5, label=ir)
            elif not self._plot_alt_angle:
                legend_patches = list()
                for ir in roi_x:
                    color_base = self.standard_cmap[(ir-1) % self.standard_cmap_n_colors]
                    alpha = np.array([1 - (self._roi_ias[ir] / self.ref_max_angle)]).T
                    alpha[alpha < 0] = 0
                    color_map = np.concatenate((color_base * np.ones([len(self._roi_ias[ir]), 1]), alpha), axis=1)
                    pp.scatter(x=roi_x[ir]-min_x, y=roi_y[ir]-min_y, s=0.5, c=color_map)
                    legend_patches.append(mpatches.Patch(color=color_base, label=ir))
                pp.legend(handles=legend_patches)
            else:
                cmap_base = matplotlib.colormaps["seismic"]     # 2026 port: accessing colormaps as pyplot attributes was deprecated
                for ir in roi_x:
                    alpha = 1 - (self._roi_ias[ir] / self.ref_max_angle)
                    alpha[alpha < 0] = 0
                    c = np.rad2deg(self._roi_alt_ias[ir])
                    c_norm = c.copy()
                    c_norm[c_norm < -self.ref_max_alt_angle_deg] = -self.ref_max_alt_angle_deg
                    c_norm[c_norm > self.ref_max_alt_angle_deg] = self.ref_max_alt_angle_deg
                    cmap_positions = np.round(cmap_base.N * (c_norm - (-self.ref_max_alt_angle_deg)) / (2 * self.ref_max_alt_angle_deg)).astype("int")
                    c_norm_cmap = cmap_base(cmap_positions)
                    c_norm_cmap[:, -1] = alpha
                    pp.scatter(x=roi_x[ir]-min_x, y=roi_y[ir]-min_y, s=2, c=c_norm_cmap)

            pp.axis("tight")
            pp.title(self._title, fontweight="bold")
            pp.xlabel("Long", fontweight="bold")
            pp.xticks(labels_x_pos, labels_x)
            pp.ylabel("Lat", fontweight="bold")
            pp.yticks(labels_y_pos, labels_y)
            self.show(explicit=False)
