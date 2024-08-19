#! /usr/bin/env python3

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

import sys
import rospy
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TwistWithCovarianceStamped
from rave.helper import Helper
from rave.outlier_rejection import OutlierRejectionClass
from rave.inlier_estimation import InlierEstimationClass


class VelEstimator:
    """
    Velocity estimator class
    """
    def __init__(self):
        # Topic names
        radar_topic = rospy.get_param("radar_topic")
        pub_name = rospy.get_param("pub_name")

        # Publishers and subscribers
        self.pub_ = rospy.Publisher(pub_name, TwistWithCovarianceStamped, queue_size=10)
        self.sub_ = rospy.Subscriber(radar_topic, PointCloud2, self.RadarCallback)

        self.runtimes = []
        self.window_rejection = []

        self.window_size = rospy.get_param("window_size")
        self.type_dataset = rospy.get_param("type_dataset")

        # Initial filtering parameters
        min_range = rospy.get_param("min_range")
        max_range = rospy.get_param("max_range")
        elevation_threshold = rospy.get_param("elevation_threshold_deg")
        azimuth_threshold = rospy.get_param("azimuth_threshold_deg")
        min_snr = rospy.get_param("min_snr")

        # Zero velocity detection parameters
        outlier_per_zero = rospy.get_param("outlier_percentage")
        zero_vel_threshold = rospy.get_param("zero_velocity_threshold")

        # Consecutive difference threshold
        self.max_a = rospy.get_param("max_a")
        self.mean_filter_threshold = rospy.get_param("mean_filter_threshold")

        # RANSAC parameters
        self.outlier_probability = rospy.get_param("outlier_probability")
        self.success_probability = rospy.get_param("success_probability")

        # GNC parameters
        self.doppler_sigma = rospy.get_param("doppler_sigma")
        self.percentage_inliers_gnc = rospy.get_param("percentage_inliers_gnc")

        self.estimate_cov = rospy.get_param("estimate_cov")

        # Initialize helper class
        self.helper = Helper(
            min_range,
            max_range,
            elevation_threshold,
            azimuth_threshold,
            min_snr,
            outlier_per_zero,
            zero_vel_threshold,
            self.type_dataset,
        )
        self.v_ = [0, 0, 0]
        self.cov_ = np.zeros((3, 3))
        self.last_v_ = [0, 0, 0]
        self.last_time_ = 0
        self.rejected_points = 0

        self.use_consecutive_diff = rospy.get_param("use_consecutive_diff")
        self.use_zero_velocity = rospy.get_param("use_zero_velocity")
        self.sigma_zero_vec_x = rospy.get_param("sigma_zero_vec_x")
        self.sigma_zero_vec_y = rospy.get_param("sigma_zero_vec_y")
        self.sigma_zero_vec_z = rospy.get_param("sigma_zero_vec_z")

        self.inliers_estimation_method = rospy.get_param("inliers_estimation_method")
        self.outlier_rejection_method = rospy.get_param("outlier_rejection_method")

        self.inlier_threshold_ = rospy.get_param("inlier_threshold")

        self.iteration_mlesac = rospy.get_param("iteration_mlesac")
        self.converge_thres = rospy.get_param("converge_thres")
        self.sigma_vr_mlesac = rospy.get_param("sigma_vr_mlesac")

        self.k_ = rospy.get_param("k")
        self.norm_method_ = rospy.get_param("norm_method")
        self.barron_a_ = rospy.get_param("barron_a")
        self.barron_c_ = rospy.get_param("barron_c")

        self.parameter_dict = {
            "min_range": min_range,
            "max_range": max_range,
            "elevation_threshold": elevation_threshold,
            "azimuth_threshold": azimuth_threshold,
            "min_snr": min_snr,
            "outlier_per_zero": outlier_per_zero,
            "zero_vel_threshold": zero_vel_threshold,
            "estimate_cov": self.estimate_cov,
            "max_a": self.max_a,
            "mean_filter_threshold": self.mean_filter_threshold,
            "outlier_probability": self.outlier_probability,
            "success_probability": self.success_probability,
            "doppler_sigma": self.doppler_sigma,
            "use_consecutive_diff": self.use_consecutive_diff,
            "use_zero_velocity": self.use_zero_velocity,
            "percentage_inliers_gnc": self.percentage_inliers_gnc,
            "iteration_mlesac": self.iteration_mlesac,
            "converge_thres": self.converge_thres,
            "sigma_vr_mlesac": self.sigma_vr_mlesac,
            "type_dataset": self.type_dataset,
            "outlier_rejection_method": self.outlier_rejection_method,
            "inliers_estimation_method": self.inliers_estimation_method,
            "k": self.k_,
            "norm_method": self.norm_method_,
            "barron_a": self.barron_a_,
            "barron_c": self.barron_c_,
            "inlier_threshold": self.inlier_threshold_,
            "window_size": self.window_size,
        }
        self.outlier_rejection_ = OutlierRejectionClass(self.parameter_dict)
        self.inlier_est_ = InlierEstimationClass(self.parameter_dict)

    def RadarCallback(self, radar_msg):
        """
        Callback function for the radar data
        """
        if self.type_dataset == "IRS":
            pts_list = list(
                pc2.read_points(
                    radar_msg,
                    field_names=[
                        "x",
                        "y",
                        "z",
                        "snr_db",
                        "v_doppler_mps",
                        "noise_db",
                        "range",
                    ],
                )
            )
        elif self.type_dataset == "VOD":
            pts_list = list(
                pc2.read_points(
                    radar_msg,
                    field_names=["x", "y", "z", "snr_db", "noise_db", "v_doppler_mps"],
                )
            )
        else:  # Coloradar dataset
            pts_list = list(
                pc2.read_points(
                    radar_msg,
                    field_names=["x", "y", "z", "intensity", "range", "doppler"],
                )
            )

        pts = np.array(pts_list, dtype=np.float32)
        time_ = radar_msg.header.stamp
        print("\n")
        rospy.loginfo("Pointcloud received")
        self.FilterPoints(pts, time_)

    def FilterPoints(self, pts, time_):
        """
        Filtering the pointcloud based on the range, elevation, azimuth and intensity data
        """

        rospy.loginfo("Filtering points")
        Ntargets = pts.shape[0]
        print("Pointcloud has ", Ntargets, " targets")

        data_XYZI = np.column_stack((pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]))

        idx_filtered = self.helper.XYZI_filtering(data_XYZI)
        Ntargets_valid = idx_filtered.shape[0]

        print("Fitlered pointcloud has ", Ntargets_valid, " valid targets")

        if self.type_dataset == "IRS":
            radar_doppler = pts[idx_filtered, 4]
            radar_x = pts[idx_filtered, 0]
            radar_y = pts[idx_filtered, 1]
            radar_z = pts[idx_filtered, 2]
        elif self.type_dataset == "VOD":
            radar_doppler = pts[idx_filtered, 5]
            radar_x = pts[idx_filtered, 0]
            radar_y = pts[idx_filtered, 1]
            radar_z = pts[idx_filtered, 2]
        else:  # Coloradar dataset
            radar_doppler = pts[idx_filtered, 5]
            radar_x = -pts[idx_filtered, 1]
            radar_y = pts[idx_filtered, 0]
            radar_z = pts[idx_filtered, 2]

        radar_snr = pts[idx_filtered, 3]
        radar_range = np.sqrt(radar_x**2 + radar_y**2 + radar_z**2)

        radar_rx = radar_x / radar_range
        radar_ry = radar_y / radar_range
        radar_rz = radar_z / radar_range

        if self.use_zero_velocity:
            if self.helper.ZeroVelocityDetection(radar_doppler):
                v_ = [0, 0, 0]
                self.cov_ = np.diag(
                    [
                        np.power(self.sigma_zero_vec_x, 2),
                        np.power(self.sigma_zero_vec_y, 2),
                        np.power(self.sigma_zero_vec_z, 2),
                    ]
                )
                print("Zero velocity detected")
                self.last_v_ = v_
                self.PublishVelocity(v_, time_)
            else:
                print("Zero velocity not detected")
                if Ntargets_valid < 3:
                    print("Not enough points to estimate velocity")
                else:
                    if self.EstimateVelocity(
                        radar_doppler, radar_snr, radar_rx, radar_ry, radar_rz
                    ):
                        print("Time", time_)
                        if self.use_consecutive_diff:
                            pass_ = True

                            if len(self.window_rejection) < self.window_size:
                                self.window_rejection.append((self.v_))
                                pass_ = True
                                self.PublishVelocity(self.v_, time_)
                                self.last_v_ = self.v_
                                self.last_time_ = time_.to_sec()
                            else:
                                if (
                                    np.linalg.norm(
                                        (self.v_) - np.mean(self.window_rejection)
                                    )
                                    < self.mean_filter_threshold
                                ):
                                    pass_ = True
                                else:
                                    pass_ = False

                                if pass_:
                                    if (
                                        np.linalg.norm(
                                            np.array(self.v_) - np.array(self.last_v_)
                                        )
                                        / (time_.to_sec() - self.last_time_)
                                    ) < self.max_a:
                                        self.window_rejection.pop(0)
                                        self.window_rejection.append((self.v_))
                                        self.PublishVelocity(self.v_, time_)
                                        self.last_v_ = self.v_
                                        self.last_time_ = time_.to_sec()
                                    else:
                                        print("Consecutive difference too high")
                                        self.rejected_points += 1

                                else:
                                    print("Mean filter threshold not satisfied")
                                    self.last_time_ = time_.to_sec()
                                    self.rejected_points += 1

                        else:
                            self.PublishVelocity(self.v_, time_)

        else:
            if Ntargets_valid < 3:
                print("Not enough points to estimate velocity")
            else:
                if self.EstimateVelocity(
                    radar_doppler, radar_snr, radar_rx, radar_ry, radar_rz
                ):
                    print("Time", time_.to_sec())
                    if self.use_consecutive_diff:
                        if (
                            np.linalg.norm(np.array(self.v_) - np.array(self.last_v_))
                            / (time_.to_sec() - self.last_time_)
                        ) < self.max_a:
                            self.PublishVelocity(self.v_, time_)
                            self.last_v_ = self.v_
                            self.last_time_ = time_.to_sec()
                        else:
                            print("Consecutive difference too high")

                    else:
                        self.PublishVelocity(self.v_, time_)

    def EstimateVelocity(self, radar_doppler, radar_snr, radar_rx, radar_ry, radar_rz):
        """
        Estimate the ego-velocity of the radar
        """
        print("Estimating velocity")
        t_before = rospy.get_time()

        if radar_doppler.shape[0] < 3:
            print("Not enough points to estimate velocity")
            return False
        else:
            if self.estimate_cov:
                best_inliers, self.v_, self.cov_ = (
                    self.outlier_rejection_.RejectOutliers(
                        radar_doppler,
                        radar_rx,
                        radar_ry,
                        radar_rz,
                        radar_snr,
                        self.last_v_,
                    )
                )
            else:
                best_inliers, self.v_ = self.outlier_rejection_.RejectOutliers(
                    radar_doppler, radar_rx, radar_ry, radar_rz, radar_snr, self.last_v_
                )

            doppler_i = radar_doppler[best_inliers]
            rx_i = radar_rx[best_inliers]
            ry_i = radar_ry[best_inliers]
            rz_i = radar_rz[best_inliers]
            snr_i = radar_snr[best_inliers]

            if self.estimate_cov:
                self.v_, self.cov_ = self.inlier_est_.EstimateVelocityInliers(
                    doppler_i, rx_i, ry_i, rz_i, snr_i, self.v_, self.cov_
                )
            else:
                self.v_ = self.inlier_est_.EstimateVelocityInliers(
                    doppler_i, rx_i, ry_i, rz_i, snr_i, self.v_, self.cov_
                )

        t_after = rospy.get_time()
        print("Time for estimation: ", t_after - t_before)
        self.runtimes.append(t_after - t_before)
        return True

    def PublishVelocity(self, v_, time_):
        """
        Publish the estimated velocity
        """
        print("Publishing velocity")
        print("Average runtime: ", np.mean(self.runtimes))
        twist_msg = TwistWithCovarianceStamped()
        twist_msg.header.stamp = time_
        twist_msg.twist.twist.linear.x = v_[0]
        twist_msg.twist.twist.linear.y = v_[1]
        twist_msg.twist.twist.linear.z = v_[2]

        if self.estimate_cov:
            twist_msg.twist.covariance[0] = self.cov_[0, 0]
            twist_msg.twist.covariance[1] = self.cov_[0, 1]
            twist_msg.twist.covariance[2] = self.cov_[0, 2]
            twist_msg.twist.covariance[6] = self.cov_[1, 0]
            twist_msg.twist.covariance[7] = self.cov_[1, 1]
            twist_msg.twist.covariance[8] = self.cov_[1, 2]
            twist_msg.twist.covariance[12] = self.cov_[2, 0]
            twist_msg.twist.covariance[13] = self.cov_[2, 1]
            twist_msg.twist.covariance[14] = self.cov_[2, 2]

        print("Number of rejected points: ", self.rejected_points)

        self.pub_.publish(twist_msg)
        return


def main():
    rospy.init_node("Radar ego-velocity estimation node")
    velocity_estimator = VelEstimator()

    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        sys.exit()


if __name__ == "__main__":
    main()
