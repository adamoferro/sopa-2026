#!/usr/bin/python3

#   Copyright 2016-2020 Adamo Ferro
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

from processing.simulators.sim_base import _SIMBase
from geometry.coordinate_converter import CoordinateConverter as cc
import numpy as np
from scipy.ndimage import gaussian_filter
import pandas as pd


class sfRSsimAltAngle(_SIMBase):
    """docstring for sfRSsimAltAngle."""

    def __init__(self, geom_obj, dem_obj, param_obj, n_processes=1, use_data_dt=True, simulate_chirp=True, instr_obj=None, roi_img=None):

        if roi_img is None:
            raise ValueError("ERROR: ROI image is mandatory.")

        super().__init__(geom_obj, dem_obj, param_obj, n_processes, use_data_dt, simulate_chirp, instr_obj, roi_img)
        self.n_frames_offset_for_dir_estimation = 1
        self.cos_exp = 1000
        self.alt_sim_spacing = 25
        self.act_sim_spacing = 25
        self.gaussian_filter_sigma = 16      # double wrt dRSsim because the starting grid is x2 sampled
        self.alt_sim_forward_max_distance = 1500   #500
        self.alt_sim_backward_max_distance = 1500
        self.act_sim_max_distance = 50000

        # /2 because the "middle" values are used for facet approximation
        alt_sim_range = np.arange(-self.alt_sim_backward_max_distance, self.alt_sim_forward_max_distance + self.alt_sim_spacing/2., self.alt_sim_spacing/2.)
        act_sim_range = np.arange(-self.act_sim_max_distance, self.act_sim_max_distance + self.act_sim_spacing/2., self.act_sim_spacing/2.)
        self.alt_sim_matrix = np.array([alt_sim_range]).T * np.ones([1, len(act_sim_range)])
        self.act_sim_matrix = np.array([act_sim_range]) * np.ones([len(alt_sim_range), 1])

    def simulate(self, orbit_data):
        return super().simulate(orbit_data, sfRSsimAltAngle._simulate_frame_range, further_pars=[self.cos_exp])

    @staticmethod       # defined static to allow calling from ray avoiding the "pickling" of the object
    def _simulate_frame_range(id, n_processes, orbit_xyz, frame_range, sim_n_samples, local_direction_g, UTMNorthing_g, UTMEasting_g, UTMZone_g, rxwin_ref, input_scr_data_shifts_alignment_t, dt, dem_obj, geom_obj, alt_sim_matrix, act_sim_matrix, gaussian_filter_sigma, further_pars, roi_img=None):
        cos_exp = further_pars[0]

        _C = 299792458.

        n_frames_to_be_simulated = len(frame_range)

        # frame_range=[318]

        sim_image = np.zeros([sim_n_samples, n_frames_to_be_simulated])         #, dtype="int16")   # int16 and uint8 saves memory but considerably slows down the processing as numpy cannot handle ints in a fast way
        uncert_image = np.zeros(sim_image.shape, dtype="uint8")
        ground_distance_min_image = np.nan * np.ones(sim_image.shape)
        ground_distance_max_image = np.nan * np.ones(sim_image.shape)
        first_return_lats = np.zeros(n_frames_to_be_simulated)
        first_return_lons = np.zeros(n_frames_to_be_simulated)
        roi_lats = dict()
        roi_lons = dict()
        roi_ias = dict()
        roi_alt_ias = dict()
        pd_hist_argmax_total = pd.DataFrame()

        status_str_pre = '\t'+"|\t\t"*id
        status_str_post = "\t\t|"*(n_processes-id-1)
        last_status = -1
        for count, i_frame_tbs in enumerate(frame_range):
            if count == n_frames_to_be_simulated-1:
                new_status = 100
            else:
                new_status = int(count / n_frames_to_be_simulated * 20) * 5
            if new_status != last_status:
                print(status_str_pre + '{:>2}'.format(new_status) + status_str_post)
                last_status = new_status

            local_direction = local_direction_g[i_frame_tbs]
            UTMNorthing = UTMNorthing_g[i_frame_tbs]
            UTMEasting = UTMEasting_g[i_frame_tbs]
            UTMZone = UTMZone_g[i_frame_tbs]
            sc_x_cart_local = orbit_xyz[0][i_frame_tbs]
            sc_y_cart_local = orbit_xyz[1][i_frame_tbs]
            sc_z_cart_local = orbit_xyz[2][i_frame_tbs]
            sc_cart_vector = np.array([[[sc_x_cart_local]], [[sc_y_cart_local]], [[sc_z_cart_local]]])
            sc_cart_unit_vector = cc.unit_vector(sc_cart_vector)

            # along-track unit vector (approximation, TODO improve using only tangential part)
            if i_frame_tbs != len(orbit_xyz[0])-1:
                alt_unit_x = orbit_xyz[0][i_frame_tbs+1] - orbit_xyz[0][i_frame_tbs]
                alt_unit_y = orbit_xyz[1][i_frame_tbs+1] - orbit_xyz[1][i_frame_tbs]
                alt_unit_z = orbit_xyz[2][i_frame_tbs+1] - orbit_xyz[2][i_frame_tbs]
                alt_unit_vector = cc.unit_vector(np.array([[[alt_unit_x]], [[alt_unit_y]], [[alt_unit_z]]]))
                # act_unit_vector = np.cross(alt_unit_vector, sc_cart_unit_vector, axis=0)
            else:
                pass    # do nothing, use the old one (TODO improve this!)

            UTMNorthing_surface_matrix = UTMNorthing + (alt_sim_matrix * np.sin(local_direction) - act_sim_matrix * np.cos(local_direction))
            UTMEasting_surface_matrix = UTMEasting + (alt_sim_matrix * np.cos(local_direction) + act_sim_matrix * np.sin(local_direction))
            Lat_surface_matrix, Long_surface_matrix = geom_obj.UTM_to_LL(UTMNorthing_surface_matrix, UTMEasting_surface_matrix, UTMZone)

            min_lat = np.min(Lat_surface_matrix)
            max_lat = np.max(Lat_surface_matrix)
            min_long = np.min(Long_surface_matrix)
            max_long = np.max(Long_surface_matrix)

            # create an upscaled interpolated floating-point dem, to avoid spurious "steps" in the calculation of facet angles
            sub_dem_obj = dem_obj.get_scaled_subdem_from_latlon_limits(max_lat, min_lat, max_long, min_long, scale_factor=16)
            dem_surface_radius_matrix = sub_dem_obj.get_dem_radius_from_lat_lon(Lat_surface_matrix, Long_surface_matrix)

            # import matplotlib
            # matplotlib.use("WebAgg")
            # from matplotlib import pyplot as pp
            # pp.plot(dem_surface_radius_matrix[0, :])
            # # pp.plot(sub_dem_obj.dem[0, :])
            # pp.show()
            # exit()
            # # print(dem_surface_radius_matrix.min())
            # # print(dem_surface_radius_matrix.max())
            # # exit()


            # import matplotlib
            # matplotlib.use("WebAgg")
            # from matplotlib import pyplot as pp
            # pp.figure(figsize=(18, 8))
            # pp.imshow(gaussian_filter(dem_surface_radius_matrix, gaussian_filter_sigma), vmin=3374798, vmax=3374956)
            # # pp.figure(figsize=(18, 8))
            # # pp.imshow(dem_obj.get_dem_radius_from_lat_lon(Lat_surface_matrix, Long_surface_matrix))
            # pp.show()
            # exit()


            # smooth the DEM to avoid "stripes" in the simulation, avoiding also holes which may cause artifacts in the simulation
            dummy_check_matrix = (dem_surface_radius_matrix == sub_dem_obj.dummy_value)
            if dummy_check_matrix.any():
                uncert_image[:, count] = 1

                # selective gaussian smoothing based on the idea reported by the user David here: https://stackoverflow.com/a/36307291
                # used here to keep consistent facets later. This was not an issue in dRSsim.
                tmp1 = dem_surface_radius_matrix.copy()
                tmp1[dummy_check_matrix] = 0
                tmp1_filtered = gaussian_filter(tmp1, gaussian_filter_sigma)
                tmp2 = np.ones(dem_surface_radius_matrix.shape)
                tmp2[dummy_check_matrix] = 0
                tmp2_filtered = gaussian_filter(tmp2, gaussian_filter_sigma)
                tmp2_filtered[tmp1_filtered == 0] = 1
                dem_surface_radius_matrix = tmp1_filtered/tmp2_filtered
                dem_surface_radius_matrix[dummy_check_matrix] = np.nan             # radius too far for being recorded in the simulation, but different from 0 to avoid numerical problems later
            else:
                dem_surface_radius_matrix = gaussian_filter(dem_surface_radius_matrix, gaussian_filter_sigma)

            # FACET APPROACH: the vertical and horizontal inclination of each facet is
            #   approximated by the vector joining the middle points of the vertical
            #   and horizontal sides of the squares. The center of the squares is used
            #   to calculate the distance from the spacecraft.
            # useful views
            act_middle_UTMNorthing_surface_matrix = UTMNorthing_surface_matrix[0::2, 1::2]
            act_middle_UTMEasting_surface_matrix = UTMEasting_surface_matrix[0::2, 1::2]
            act_middle_radius_surface_matrix = dem_surface_radius_matrix[0::2, 1::2]

            alt_middle_UTMNorthing_surface_matrix = UTMNorthing_surface_matrix[1::2, 0::2]
            alt_middle_UTMEasting_surface_matrix = UTMEasting_surface_matrix[1::2, 0::2]
            alt_middle_radius_surface_matrix = dem_surface_radius_matrix[1::2, 0::2]

            center_UTMNorthing_surface_matrix = UTMNorthing_surface_matrix[1::2, 1::2]
            center_UTMEasting_surface_matrix = UTMEasting_surface_matrix[1::2, 1::2]
            center_radius_surface_matrix = dem_surface_radius_matrix[1::2, 1::2]
            ground_distance_from_nadir = np.sqrt((center_UTMNorthing_surface_matrix-UTMNorthing)**2 + (center_UTMEasting_surface_matrix-UTMEasting)**2)

            # conversion to lat-long and cartesian
            act_middle_Lat_surface_matrix, act_middle_Long_surface_matrix = geom_obj.UTM_to_LL(act_middle_UTMNorthing_surface_matrix, act_middle_UTMEasting_surface_matrix, UTMZone)
            act_middle_x_cart_surface_matrix, act_middle_y_cart_surface_matrix, act_middle_z_cart_surface_matrix = geom_obj.to_xyz_from_latlon_and_radius(act_middle_Lat_surface_matrix, act_middle_Long_surface_matrix, act_middle_radius_surface_matrix)

            alt_middle_Lat_surface_matrix, alt_middle_Long_surface_matrix = geom_obj.UTM_to_LL(alt_middle_UTMNorthing_surface_matrix, alt_middle_UTMEasting_surface_matrix, UTMZone)
            alt_middle_x_cart_surface_matrix, alt_middle_y_cart_surface_matrix, alt_middle_z_cart_surface_matrix = geom_obj.to_xyz_from_latlon_and_radius(alt_middle_Lat_surface_matrix, alt_middle_Long_surface_matrix, alt_middle_radius_surface_matrix)

            center_Lat_surface_matrix, center_Long_surface_matrix = geom_obj.UTM_to_LL(center_UTMNorthing_surface_matrix, center_UTMEasting_surface_matrix, UTMZone)
            center_x_cart_surface_matrix, center_y_cart_surface_matrix, center_z_cart_surface_matrix = geom_obj.to_xyz_from_latlon_and_radius(center_Lat_surface_matrix, center_Long_surface_matrix, center_radius_surface_matrix)

            # creation of 3D vector matrices
            center_stack = np.stack((center_x_cart_surface_matrix, center_y_cart_surface_matrix, center_z_cart_surface_matrix))
            v_stack = np.stack((act_middle_x_cart_surface_matrix, act_middle_y_cart_surface_matrix, act_middle_z_cart_surface_matrix))
            h_stack = np.stack((alt_middle_x_cart_surface_matrix, alt_middle_y_cart_surface_matrix, alt_middle_z_cart_surface_matrix))

            # vectors describing the two approximated axes of each facet
            v_stack_diffs = v_stack[:, 1:, :] - v_stack[:, :-1, :]
            h_stack_diffs = -(h_stack[:, :, 1:] - h_stack[:, :, :-1])

            # calculate the approximate area of each facet as the product of the length of the facet "axes" vectors
            v_stack_diffs_norms = np.linalg.norm(v_stack_diffs, axis=0)
            h_stack_diffs_norms = np.linalg.norm(h_stack_diffs, axis=0)
            facet_approx_areas = v_stack_diffs_norms * h_stack_diffs_norms

            # vectors connecting the spacecraft to the center of each facet and their length
            surface_to_sc_vectors = sc_cart_vector*np.ones(center_x_cart_surface_matrix.shape) - center_stack
            d_surface = np.linalg.norm(surface_to_sc_vectors, axis=0)

            # time to facet from spacecraft, and conversion to radargram sample (range) position
            t_surface = 2. * d_surface / _C
            sample_pos_surface = np.round(((t_surface - rxwin_ref + input_scr_data_shifts_alignment_t[i_frame_tbs]) / dt)).astype("int")        # use rxwin_ref instead of input_rxwin_data_t_local

            # set out-of-range sample positions to the farthest position (useful later for detecting the first-return position)
            sample_pos_surface[(sample_pos_surface < 0) | (sample_pos_surface >= sim_n_samples)] = sim_n_samples

            # find first return coordinates
            first_return_ids = np.unravel_index(np.argmin(sample_pos_surface), sample_pos_surface.shape)
            first_return_lats[count] = center_Lat_surface_matrix[first_return_ids]
            first_return_lons[count] = center_Long_surface_matrix[first_return_ids]

            # vectors describing the normal of each facet
            normal_surface_vectors = np.cross(v_stack_diffs, h_stack_diffs, axis=0)
            normal_surface_vectors = normal_surface_vectors / np.linalg.norm(normal_surface_vectors, axis=0)

            # angle between the two previous vectors (0=facet facing the spacecraft, pi/2=facet oriented perpendicularly to the spacecraft)
            angle_between_sc_and_normal_surface = geom_obj.angle_between(normal_surface_vectors, surface_to_sc_vectors)

            # calculation of the angle between the SC and the facet, projected on the along-track plane
            alt_dotp = np.tensordot(normal_surface_vectors, alt_unit_vector, axes=(0, 0))[:, :, 0, 0]
            # act_dotp = np.tensordot(normal_surface_vectors, act_unit_vector, axes=(0, 0))[:, :, 0, 0]
            r_dotp = np.tensordot(normal_surface_vectors, sc_cart_unit_vector, axes=(0, 0))[:, :, 0, 0]
            alt_angle = -np.arctan2(alt_dotp, r_dotp)

            facet_approx_areas = facet_approx_areas.flatten()
            ground_distance_from_nadir = ground_distance_from_nadir.flatten()
            sample_pos_surface = sample_pos_surface.flatten()
            roi_sample_pos = np.where(roi_img[:, i_frame_tbs])
            sample_pos_surface_ok_ids = np.isin(sample_pos_surface, roi_sample_pos)     # select only the samples that have a correspondence in the ROI image
            sample_pos_surface = sample_pos_surface[sample_pos_surface_ok_ids]

            facet_approx_areas = facet_approx_areas[sample_pos_surface_ok_ids]
            angle_between_sc_and_normal_surface = angle_between_sc_and_normal_surface.flatten()
            angle_between_sc_and_normal_surface = angle_between_sc_and_normal_surface[sample_pos_surface_ok_ids]
            alt_angle = alt_angle.flatten()[sample_pos_surface_ok_ids]

            ia_angle_lists = [list() for il in range(sim_n_samples)]
            alt_angle_lists = [list() for il in range(sim_n_samples)]       # TODO can be improved by pre-counting the max number of repetitions and creating the pd.DataFrame directly?
            for ip, sp in enumerate(sample_pos_surface):
                ia_angle_lists[sp].append(angle_between_sc_and_normal_surface[ip])
                alt_angle_lists[sp].append(alt_angle[ip])

            ia_angle_limit = np.deg2rad(2)
            alt_angle_limit = np.deg2rad(2)
            alt_angle_limit_min = np.deg2rad(0.25)

            pd_ia_angle_lists = pd.DataFrame(ia_angle_lists)
            pd_alt_angle_lists = pd.DataFrame(alt_angle_lists)

            # first reduction, eliminating only-nan rows (note: for each ia, an alt_angle is defined as well)
            sel_rows = pd_ia_angle_lists.notna().any(axis=1)
            pd_ia_angle_lists = pd_ia_angle_lists[sel_rows]
            pd_alt_angle_lists = pd_alt_angle_lists[sel_rows]

            # select only rows where at least one global ia is small
            pd_ia_angle_lists[np.abs(pd_ia_angle_lists) > ia_angle_limit] = np.nan
            # sel_rows = (np.abs(pd_ia_angle_lists) <= ia_angle_limit).any(axis=1)
            sel_rows = pd_ia_angle_lists.notna().any(axis=1)
            pd_ia_angle_lists = pd_ia_angle_lists[sel_rows]
            pd_alt_angle_lists = pd_alt_angle_lists[sel_rows]

            # select only rows where at least one alt angle is small
            pd_alt_angle_lists[np.abs(pd_alt_angle_lists) > alt_angle_limit] = np.nan
            pd_alt_angle_lists[np.abs(pd_alt_angle_lists) < alt_angle_limit_min] = np.nan
            # sel_rows = (np.abs(pd_alt_angle_lists) <= alt_angle_limit).any(axis=1)
            sel_rows = pd_alt_angle_lists.notna().any(axis=1)
            pd_ia_angle_lists = pd_ia_angle_lists[sel_rows]
            pd_alt_angle_lists = pd_alt_angle_lists[sel_rows]

            # print(pd_alt_angle_lists.count(axis=1).sort_values())
            # # exit()

            pd_alt_angle_lists = np.rad2deg(pd_alt_angle_lists)

            bins = np.arange(-1.825, 1.875, 0.05)

            pd_hist_argmax = (pd_alt_angle_lists.apply(lambda a: np.histogram(a, bins=bins)[0].argmax(), axis=1)).to_frame("argmax")

            # print(pd_hist_argmax)
            # exit()

            # x, be = np.histogram(np.rad2deg(pd_alt_angle_lists.loc[392]), )       #.hist(bins=np.arange(-1.5, 1.55, 0.05))
            # bc = (be[1:] + be[:-1])/2
            # x_g = gaussian_filter(x, 2)
            #
            # print(x_g.argmax())
            # print(x_g.argmax()*.05-1.8)
            #
            # # print(x.shape)
            # # print(1.8*2/.05+1)
            # # exit()
            #
            # import matplotlib
            # matplotlib.use("WebAgg")
            # from matplotlib import pyplot as pp
            #
            # fig = pp.figure(figsize=(18, 8))
            # ax = fig.gca()
            #
            # # print(x)
            # # np.rad2deg(pd_alt_angle_lists.loc[392]).hist(bins=np.arange(-1.825, 1.875, 0.05), ax=ax)
            #
            # pp.plot(bc, x_g)
            # pp.xlim(-2, 2)
            # pp.show()
            # exit()

            pd_hist_argmax["count"] = pd_alt_angle_lists.count(axis=1)
            pd_hist_argmax["sample"] = pd_alt_angle_lists.index.astype(int)
            pd_hist_argmax["frame"] = i_frame_tbs

            # print(pd_hist_argmax)
            # exit()

            pd_hist_argmax_total = pd.concat([pd_hist_argmax_total, pd_hist_argmax], sort=False)        # 2026 port: DataFrame.append was removed in pandas 2.0

            # print(pd_hist_argmax_total)
            # #
            # if count == 1:
            #     exit()
            # #     # print(pd_alt_angle_lists)
            # #     print(pd_alt_angle_lists_total)
            #     print(pd_alt_angle_lists_total["frame"])
            #     print(pd_alt_angle_lists_total[pd_alt_angle_lists_total["frame"] == 2])
            # #     # print(pd_alt_angle_lists_total.columns)
            # #     # print(pd_alt_angle_lists_total.shape)
            #     exit()

            # if cos_exp != 0:
            #     np.add.at(sim_image[:, count], sample_pos_surface, np.cos(angle_between_sc_and_normal_surface)**cos_exp * facet_approx_areas)
            # else:
            #     np.add.at(sim_image[:, count], sample_pos_surface, facet_approx_areas)
            #
            # ground_distance_from_nadir = ground_distance_from_nadir[sample_pos_surface_ok_ids]
            #
            # # SEE COMMENTS IN DRSSIM FOR THE FOLLOWING CODE
            # sorting_indexes = np.argsort(ground_distance_from_nadir)
            # ground_distance_from_nadir = ground_distance_from_nadir[sorting_indexes]
            # sample_pos_surface = sample_pos_surface[sorting_indexes]
            #
            # unique_sample_pos_surface, unique_sample_pos_surface_ids = np.unique(sample_pos_surface, return_index=True)
            # unique_ground_distance_from_nadir = ground_distance_from_nadir[unique_sample_pos_surface_ids]
            # ground_distance_min_image[unique_sample_pos_surface, count] = unique_ground_distance_from_nadir
            #
            # sample_pos_surface = np.flip(sample_pos_surface)
            # ground_distance_from_nadir = np.flip(ground_distance_from_nadir)
            # unique_sample_pos_surface, unique_sample_pos_surface_ids = np.unique(sample_pos_surface, return_index=True)
            # unique_ground_distance_from_nadir = ground_distance_from_nadir[unique_sample_pos_surface_ids]
            # ground_distance_max_image[unique_sample_pos_surface, count] = unique_ground_distance_from_nadir

        # sim_image[np.isnan(sim_image)] = 0
        # return (sim_image, uncert_image, first_return_lats, first_return_lons, ground_distance_min_image, ground_distance_max_image, (roi_lats, roi_lons, roi_ias, roi_alt_ias))
        return (sim_image, None, None, None, None, None, (None, None, None, None, pd_hist_argmax_total))
