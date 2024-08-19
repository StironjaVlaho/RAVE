
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


class InlierEstimationClass:
    """
    Inlier estimation class for estimating the velocity of the inlier targets
    """
    def __init__(self, param_dict):
        self.param_dict_ = param_dict
        self.norm_ = NormClass(param_dict)
        pass

    def EstimateVelocityInliers(self, doppler, rx, ry, rz, snr, vel_, cov_):
        """
        Main function for estimating the velocity of the inlier targets
        """
        method = self.param_dict_["inliers_estimation_method"]
        if method == "LS":
            return self.LS(doppler, rx, ry, rz, cov_)
        elif method == "WLS":
            return self.WLS(doppler, snr, rx, ry, rz, cov_)
        elif method == "TLS":
            return self.TLS(doppler, snr, rx, ry, rz, cov_)
        elif method == "WTLS":
            return self.WTLS(doppler, snr, rx, ry, rz, cov_)
        elif method == "None":
            if self.param_dict_["estimate_cov"]:
                return vel_, cov_
            else:
                return vel_
        elif method == "OPT":
            return self.Optimization(doppler, snr, rx, ry, rz, vel_, cov_)
        else:
            print("Invalid method")
            return vel_

    def Optimization(self, doppler, snr, rx, ry, rz, last_v, cov_):
        """
        Optimization function for estimating the velocity of the inlier targets
        """
        data = (rx, ry, rz, doppler, snr)

        res = minimize(self.CostFunction, last_v, args=data, method="BFGS")
        print("Velocity estimate: ", res.x[:3])
        v_ = res.x[:3]

        p = 3
        Ntargets = snr.shape[0]
        A = np.zeros((Ntargets, p), dtype=np.float32)
        b = np.zeros((Ntargets,), dtype=np.float32)

        for i in range(len(rx)):
            A[i, :] = np.array([rx[i], ry[i], rz[i]])
            b[i] = -doppler[i]

        sigma_2 = np.sum((b - np.dot(A, v_)) ** 2) / (Ntargets - p)
        cov_ = np.linalg.inv(np.dot(A.T, A)) * sigma_2

        if self.param_dict_["estimate_cov"]:
            return v_, cov_
        else:
            return v_

    def CostFunction(self, x, rx, ry, rz, doppler, snr):
        """
        Define the cost function for the optimization
        """
        radar_doppler = -doppler

        sum_loss = 0

        for i in range(len(rx)):
            M = np.array([rx[i], ry[i], rz[i]])
            diff = radar_doppler[i] - np.dot(M, x)
            sum_loss = sum_loss + np.sum(self.norm_.Norm(diff))
        return sum_loss

    def LS(self, doppler, rx, ry, rz, cov_):
        """
        Least squares estimation for estimating the velocity of the inlier targets
        """
        Ntargets = doppler.shape[0]
        p = 3

        A = np.zeros((Ntargets, p), dtype=np.float32)
        b = np.zeros((Ntargets,), dtype=np.float32)

        for i in range(Ntargets):
            A[i, :] = np.array([rx[i], ry[i], rz[i]])

            b[i] = -doppler[i]

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
            sigma_2 = np.sum((b - np.dot(A, v_)) ** 2) / (Ntargets - p)
            cov_ = np.linalg.inv(np.dot(A.T, A)) * sigma_2

            if self.param_dict_["estimate_cov"]:
                return v_, cov_
            else:
                return v_

    def WLS(self, doppler, snr, rx, ry, rz, cov_):
        """
        Weighted least squares estimation for estimating the velocity of the inlier targets
        """ 

        Ntargets = doppler.shape[0]
        p = 3

        average_snr = np.mean(snr)
        A = np.zeros((Ntargets, p), dtype=np.float32)
        b = np.zeros((Ntargets,), dtype=np.float32)

        for i in range(Ntargets):
            A[i, :] = (snr[i] / average_snr) * np.array([rx[i], ry[i], rz[i]])

            b[i] = -(snr[i] / average_snr) * doppler[i]

        v_ = np.linalg.lstsq(A, b, rcond=1e-10)[0]
        return v_

    def TLS(self, doppler, snr, rx, ry, rz, cov_):
        """
        Total least squares estimation for estimating the velocity of the inlier targets
        """
        Ntargets = doppler.shape[0]
        p = 4

        A = np.zeros((Ntargets, p), dtype=np.float32)

        for i in range(Ntargets):
            A[i, :] = np.array([rx[i], ry[i], rz[i], -doppler[i]])

        Q, R = np.linalg.qr(A, mode="reduced")
        A = np.triu(R)

        U, S, V = np.linalg.svd(A, full_matrices=True)

        V = V.T
        solution = -V[:3, 3] / V[3, 3]

        return solution

    def WTLS(self, doppler, snr, rx, ry, rz, cov_):
        """
        Weighted total least squares estimation for estimating the velocity of the inlier targets
        """
        Ntargets = doppler.shape[0]
        p = 4

        A = np.zeros((Ntargets, p), dtype=np.float32)

        for i in range(Ntargets):
            A[i, :] = np.array([rx[i], ry[i], rz[i], -doppler[i]])

        A *= snr[:, np.newaxis] / np.mean(snr)

        Q, R = np.linalg.qr(A, mode="reduced")
        A = np.triu(R)

        U, S, V = np.linalg.svd(A, full_matrices=True)

        V = V.T
        solution = -V[:3, 3] / V[3, 3]

        return solution
