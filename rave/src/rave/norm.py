
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


class NormClass:
    """
    Normalization class for the cost function
    """
    def __init__(self, param_dict):
        self.param_dict_ = param_dict
        pass

    def Norm(self, x):
        method = self.param_dict_["norm_method"]
        if method == "Cauchy":
            return self.CauchyLoss(x, self.param_dict_["k"])
        elif method == "Huber":
            return self.HuberLoss(x, self.param_dict_["k"])
        elif method == "BarronLoss":
            return self.BarronLoss(
                x, self.param_dict_["barron_a"], self.param_dict_["barron_c"]
            )
        elif method == "None":
            return x
        else:
            print("Invalid method, choosing None")
            return x

    def CauchyLoss(self, y, c):
        return np.power(c, 2) * np.log(1 + np.power(y / c, 2))

    def HuberLoss(self, y, c):
        return np.where(np.abs(y) < c, 0.5 * np.square(y), c * (np.abs(y) - 0.5 * c))

    def BarronLoss(self, x, a, c):
        return (np.abs(a - 2) / a) * (
            np.power(((np.power(x / c, 2) / np.abs(a - 2)) + 1), a / 2) - 1
        )
