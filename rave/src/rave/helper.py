
# 
 # This file is part of the rave package.

 # Copyright (c) Vlaho-Josip Štironja, University of Zagreb Faculty of Electrical Engineering and Computing
 # 
 # This program is free software: you can redistribute it and/or modify  
 # it under the terms of the GNU General Public License as published by  
 # the Free Software Foundation, version 3.
 #
 # This program is distributed in the hope that it will be useful, but 
 # WITHOUT ANY WARRANTY; without even the implied warranty of 
 # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
 # General Public License for more details.
 #
 # You should have received a copy of the GNU General Public License 
 # along with this program. If not, see <http://www.gnu.org/licenses/>.
 #

import numpy as np
from functools import reduce


class Helper:
    """
    Helper class for filtering the data based on the range, elevation, azimuth and intensity and zero velocity detection
    """
    def __init__(
        self,
        min_range,
        max_range,
        elevation_threshold,
        azimuth_threshold,
        min_snr,
        outlier_per_zero,
        zero_vel_threshold,
        type_dataset,
    ):
        """
        Constructor
        """
        self.threshold_max_range_ = max_range
        self.threshold_min_range_ = min_range
        self.threshold_elevation_ = elevation_threshold
        self.threshold_azimuth_ = azimuth_threshold
        self.threshold_min_intensity_ = min_snr
        self.outlier_per_zero_ = outlier_per_zero
        self.zero_vel_threshold_ = zero_vel_threshold
        self.type_dataset_ = type_dataset

        pass

    def XYZI_filtering(self, data_XYZ):
        """
        Filtering the data based on the range, elevation, azimuth and intensity
        """
        if self.type_dataset_ == "IRS":
            radar_x = data_XYZ[:, 0]  # [m]
            radar_y = data_XYZ[:, 1]  # [m]
            radar_z = data_XYZ[:, 2]  # [m]

        else:
            radar_x = -data_XYZ[:, 1]  # [m]
            radar_y = data_XYZ[:, 0]  # [m]
            radar_z = data_XYZ[:, 2]  # [m]

        radar_intensity = data_XYZ[:, 3]  # [dB]
        radar_range = np.sqrt(radar_x**2 + radar_y**2 + radar_z**2)

        radar_azimuth = np.arctan2(radar_y, radar_x) - np.pi / 2
        radar_elevation = (
            np.arctan2(np.sqrt(radar_x**2 + radar_y**2), radar_z) - np.pi / 2
        )

        idx_elevation = np.nonzero(
            np.abs(radar_elevation) < np.deg2rad(self.threshold_elevation_)
        )
        idx_azimuth = np.nonzero(
            np.abs(radar_azimuth) < np.deg2rad(self.threshold_azimuth_)
        )
        idx_max = np.nonzero(radar_range < self.threshold_max_range_)
        idx_min = np.nonzero(radar_range > self.threshold_min_range_)
        idx_min_intensity = np.nonzero(radar_intensity > self.threshold_min_intensity_)

        idx_AIRE = reduce(
            np.intersect1d,
            (idx_max, idx_min, idx_azimuth, idx_elevation, idx_min_intensity),
        )
        return idx_AIRE

    def ZeroVelocityDetection(self, radar_doppler):
        """
        Zero velocity detection
        """
        median_doppler = np.median(radar_doppler)
        count = np.count_nonzero(radar_doppler == median_doppler)
        if (
            abs(median_doppler) < self.zero_vel_threshold_
            and float((len(radar_doppler) - count) / len(radar_doppler))
            < self.outlier_per_zero_
        ):
            return True
        else:
            return False
