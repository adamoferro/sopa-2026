#!/usr/bin/python3

#   Copyright 2016-2021 Adamo Ferro
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

import numpy as np
from scipy.signal import chirp


class ProcessingUtils():
    def __init__(self):
        pass

    @staticmethod
    def create_synthetic_chirp_conj(frequency, bandwidth, chirp_duration, fast_time_sampling_frequency, range_weighting_window_original, SIZE_RANGE_FFT):
        t_chirp = np.arange(0, chirp_duration, 1. / fast_time_sampling_frequency)
        range_weighting_window = np.concatenate((range_weighting_window_original, np.zeros(int(SIZE_RANGE_FFT/2))), 0)

        f0 = frequency + bandwidth/2.
        t1 = chirp_duration
        f1 = frequency - bandwidth/2.
        chirp2_real = chirp(t_chirp, f0, t1, f1)

        chirp_fft = (np.fft.fft(chirp2_real, SIZE_RANGE_FFT)) / np.sqrt(SIZE_RANGE_FFT)
        chirp_fft = chirp_fft*range_weighting_window
        chirp_fft_conj = np.conj(chirp_fft).T

        return chirp_fft_conj
