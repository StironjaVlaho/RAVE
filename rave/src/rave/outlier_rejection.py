
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
from scipy.linalg import svd
from scipy.optimize import minimize
from rave.norm import NormClass


class OutlierRejectionClass:
    """
    Outlier rejection class for rejecting the outliers
    """
    def __init__(self, param_dict):
        self.param_dict_ = param_dict
        self.norm_ = NormClass(param_dict)
        pass

    def RejectOutliers(
        self, radar_doppler, radar_rx, radar_ry, radar_rz, radar_snr, last_v
    ):
        method = self.param_dict_["outlier_rejection_method"]
        if method == "RANSAC":
            return self.RANSAC(radar_doppler, radar_rx, radar_ry, radar_rz)
        elif method == "MLESAC":
            return self.MLESAC(radar_doppler, radar_rx, radar_ry, radar_rz)
        elif method == "GNC":
            return self.GNC(radar_doppler, radar_rx, radar_ry, radar_rz)
        elif method == "OPT":
            return self.Optimization(
                radar_doppler, radar_snr, radar_rx, radar_ry, radar_rz, last_v
            )

        else:
            print("Invalid method, choosing RANSAC")
            return self.RANSAC(radar_doppler, radar_rx, radar_ry, radar_rz)

    def RANSAC(self, radar_doppler, radar_rx, radar_ry, radar_rz):
        """
        Random sample consensus (RANSAC) algorithm for outlier rejection
        """
        print("Starting RANSAC outlier rejection solver")

        ransac_iter_ = int(
            np.log(1 - self.param_dict_["success_probability"])
            / np.log(1 - np.power((1 - self.param_dict_["outlier_probability"]), 3))
        )

        best_inliers = []
        for i in range(ransac_iter_):
            idx = np.random.choice(radar_doppler.shape[0], 3, replace=False)

            radar_doppler_sample = radar_doppler[idx]
            radar_rx_sample = radar_rx[idx]
            radar_ry_sample = radar_ry[idx]
            radar_rz_sample = radar_rz[idx]

            flag = False
            if self.param_dict_["estimate_cov"]:
                current_v_, flag, cov_ = self.SolveLS(
                    radar_doppler_sample,
                    radar_rx_sample,
                    radar_ry_sample,
                    radar_rz_sample,
                )
            else:
                current_v_, flag = self.SolveLS(
                    radar_doppler_sample,
                    radar_rx_sample,
                    radar_ry_sample,
                    radar_rz_sample,
                )

            error_all = np.abs(
                np.dot(current_v_, np.array([radar_rx, radar_ry, radar_rz]))
                + radar_doppler
            )

            idx_inliers = np.nonzero(
                np.array(error_all) < self.param_dict_["inlier_threshold"]
            )

            if i == 0:
                best_inliers = idx_inliers

            if idx_inliers[0].shape[0] > best_inliers[0].shape[0]:
                best_inliers = idx_inliers

        if self.param_dict_["estimate_cov"]:
            return best_inliers, current_v_, cov_
        else:
            return best_inliers, current_v_

    def MLESAC(self, radar_doppler, radar_rx, radar_ry, radar_rz):
        """
        Maximum likelihood estimation sample consensus (MLESAC) algorithm for outlier rejection
        """
        print("Starting MLESAC outlier rejection solver")

        best_inliers = []
        dll_incr = np.inf
        iteration_mlesac = 0
        best_score = -np.inf

        while (
            np.abs(dll_incr) > self.param_dict_["converge_thres"]
            and iteration_mlesac < self.param_dict_["iteration_mlesac"]
        ):
            idx = np.random.choice(radar_doppler.shape[0], 3, replace=False)

            radar_doppler_sample = radar_doppler[idx]
            radar_rx_sample = radar_rx[idx]
            radar_ry_sample = radar_ry[idx]
            radar_rz_sample = radar_rz[idx]

            flag = False
            if self.param_dict_["estimate_cov"]:
                current_v_, flag, cov_ = self.SolveLS(
                    radar_doppler_sample,
                    radar_rx_sample,
                    radar_ry_sample,
                    radar_rz_sample,
                )
            else:
                current_v_, flag = self.SolveLS(
                    radar_doppler_sample,
                    radar_rx_sample,
                    radar_ry_sample,
                    radar_rz_sample,
                )

            error_all = np.abs(
                np.dot(current_v_, np.array([radar_rx, radar_ry, radar_rz]))
                + radar_doppler
            )

            score = (
                -1
                / (2 * np.power(self.param_dict_["sigma_vr_mlesac"], 2))
                * np.sum(np.square(error_all))
            )

            idx_inliers = np.nonzero(
                np.array(error_all) < self.param_dict_["inlier_threshold"]
            )

            if score > best_score:
                best_inliers = idx_inliers
                dll_incr = score - best_score
                best_score = score

            iteration_mlesac += 1

        if self.param_dict_["estimate_cov"]:
            return best_inliers, current_v_, cov_
        else:
            return best_inliers, current_v_

    def GNC(self, radar_doppler, radar_rx, radar_ry, radar_rz):
        """
        Graduated non-convexity (GNC) algorithm for outlier rejection
        """
        print("Starting GNC outlier rejection solver")

        weights = np.ones(len(radar_doppler))

        vel_, flag = self.SolveLS_GNC(
            weights, radar_doppler, radar_rx, radar_ry, radar_rz
        )

        res_ = self.CalculateRes(radar_doppler, radar_rx, radar_ry, radar_rz, vel_)

        c = 2 * self.param_dict_["doppler_sigma"]
        q = 4 * np.power(max(np.abs(res_)), 2) / np.square(c)

        iterator = 0
        while q > 1:
            vel_, flag = self.SolveLS_GNC(
                weights, radar_doppler, radar_rx, radar_ry, radar_rz
            )
            res_ = self.CalculateRes(radar_doppler, radar_rx, radar_ry, radar_rz, vel_)
            for i in range(len(weights)):
                weights[i] = q * np.square(c) / (np.square(res_[i]) + q * np.square(c))
            q = q / 1.4
            iterator = iterator + 1

        vel_, flag = self.SolveLS_GNC(
            weights, radar_doppler, radar_rx, radar_ry, radar_rz
        )
        res_ = self.CalculateRes(radar_doppler, radar_rx, radar_ry, radar_rz, vel_)
        print("Number of iterations: ", iterator)
        print("Weights: ", weights)

        idx = []
        for i in range(len(weights)):
            if np.square(res_[i]) < np.square(c):
                idx.append(i)

        return idx, vel_

    def Optimization(
        self, radar_doppler, radar_snr, radar_rx, radar_ry, radar_rz, last_v
    ):
        """
        Optimization function for rejecting outliers and estimating the velocity of the targets
        """
        data = (radar_rx, radar_ry, radar_rz, radar_doppler, radar_snr)

        res = minimize(self.CostFunction, last_v, args=data, method="BFGS")
        print("Velocity estimate optimzacija: ", res.x[:3])
        v_ = res.x[:3]

        error_all = np.abs(
            np.dot(v_, np.array([radar_rx, radar_ry, radar_rz])) + radar_doppler
        )
        best_inliers = np.nonzero(
            np.array(error_all) < self.param_dict_["inlier_threshold"]
        )

        p = 3
        Ntargets = radar_doppler.shape[0]
        A = np.zeros((Ntargets, p), dtype=np.float32)
        b = np.zeros((Ntargets,), dtype=np.float32)

        for i in range(len(radar_rx)):
            A[i, :] = np.array([radar_rx[i], radar_ry[i], radar_rz[i]])
            b[i] = -radar_doppler[i]

        sigma_2 = np.sum((b - np.dot(A, v_)) ** 2) / (Ntargets - p)
        cov_ = np.linalg.inv(np.dot(A.T, A)) * sigma_2

        if self.param_dict_["estimate_cov"]:
            return best_inliers, v_, cov_
        else:
            return best_inliers, v_

    def CostFunction(
        self,
        x,
        radar_rx_inliers,
        radar_ry_inliers,
        radar_rz_inliers,
        radar_doppler_inliers,
        radar_snr_inliers,
    ):
        """
        Define the cost function for the optimization
        """
        radar_rx = radar_rx_inliers
        radar_ry = radar_ry_inliers
        radar_rz = radar_rz_inliers
        radar_doppler = -radar_doppler_inliers

        sum_loss = 0

        for i in range(len(radar_rx)):
            M = np.array([radar_rx[i], radar_ry[i], radar_rz[i]])
            diff = radar_doppler[i] - np.dot(M, x)
            sum_loss = sum_loss + np.sum(self.norm_.Norm(diff))
        return sum_loss

    def SolveLS(self, radar_doppler, radar_rx_sample, radar_ry_sample, radar_rz_sample):
        """
        Least squares estimation for estimating the velocity of the inlier targets
        """

        Ntargets = radar_doppler.shape[0]
        p = 3

        A = np.zeros((Ntargets, p), dtype=np.float32)
        b = np.zeros((Ntargets,), dtype=np.float32)

        for i in range(Ntargets):
            A[i, :] = np.array(
                [radar_rx_sample[i], radar_ry_sample[i], radar_rz_sample[i]]
            )

            b[i] = -radar_doppler[i]

        HTH = np.dot(A.T, A)
        U, S, V = svd(HTH)
        if S[S.shape[0] - 1] < 1e-10:
            if self.param_dict_["estimate_cov"]:
                return np.array([0, 0, 0]), False, np.zeros((3, 3))
            else:
                return np.array([0, 0, 0]), False

        cond = S[0] / S[S.shape[0] - 1]

        if cond > 1000:
            # Matrix is ill-conditioned
            if self.param_dict_["estimate_cov"]:
                return np.array([0, 0, 0]), False, np.zeros((3, 3))
            else:
                return np.array([0, 0, 0]), False

        else:
            v_ = np.linalg.lstsq(A, b, rcond=None)[0]
            sigma_2 = np.sum((b - np.dot(A, v_)) ** 2) / (Ntargets - p)
            cov_ = np.linalg.inv(np.dot(A.T, A)) * sigma_2

            if self.param_dict_["estimate_cov"]:
                return v_, True, cov_
            else:
                return v_, True

    def SolveLS_GNC(
        self, w, radar_doppler, radar_rx_sample, radar_ry_sample, radar_rz_sample
    ):
        """
        Weighted least squares estimation for GNC
        """
        Ntargets = radar_doppler.shape[0]
        p = 3

        A = np.zeros((Ntargets, p), dtype=np.float32)
        b = np.zeros((Ntargets,), dtype=np.float32)

        for i in range(Ntargets):
            A[i, :] = w[i] * np.array(
                [radar_rx_sample[i], radar_ry_sample[i], radar_rz_sample[i]]
            )

            b[i] = -w[i] * radar_doppler[i]

        HTH = np.dot(A.T, A)
        U, S, V = svd(HTH)
        if S[S.shape[0] - 1] < 1e-10:
            return np.array([0, 0, 0]), False

        cond = S[0] / S[S.shape[0] - 1]

        if cond > 1000:
            # Matrix is ill-conditioned
            return np.array([0, 0, 0]), False

        else:
            v_ = np.linalg.lstsq(A, b, rcond=None)[0]
            return v_, True

    def CalculateRes(
        self, radar_doppler, radar_rx_sample, radar_ry_sample, radar_rz_sample, v_
    ):
        """
        Calculate the residuals for GNC
        """
        Ntargets = radar_doppler.shape[0]
        p = 3

        A = np.zeros((Ntargets, p), dtype=np.float32)
        b = np.zeros((Ntargets,), dtype=np.float32)

        for i in range(Ntargets):
            A[i, :] = np.array(
                [radar_rx_sample[i], radar_ry_sample[i], radar_rz_sample[i]]
            )

            b[i] = -radar_doppler[i]

        residuals = np.zeros((Ntargets,), dtype=np.float32)
        for i in range(Ntargets):
            residuals[i] = np.abs(radar_doppler[i] - np.dot(A[i, :], v_))
        return residuals

    # Currently in progress
    def CostFunctionAll(
        self,
        x,
        radar_rx_inliers,
        radar_ry_inliers,
        radar_rz_inliers,
        radar_doppler_inliers,
        radar_snr_inliers,
        radar_range,
    ):
        radar_rx = radar_rx_inliers
        radar_ry = radar_ry_inliers
        radar_rz = radar_rz_inliers
        radar_doppler = -radar_doppler_inliers
        sum = 0

        for i in range(len(radar_rx)):
            M = np.array(
                [
                    radar_rx[i] + x[(i + 1) * 3],
                    radar_ry[i] + x[(i + 1) * 3 + 1],
                    radar_rz[i] + x[(i + 1) * 3 + 2],
                ]
            )
            sum = (
                sum
                + np.square(radar_doppler[i] + x[(i + 1) * 3 + 3] - np.dot(M, x[:3]))
                + np.square(x[(i + 1) * 3])
                + np.square(x[(i + 1) * 3 + 1])
                + np.square(x[(i + 1) * 3 + 2])
            )
        return sum
