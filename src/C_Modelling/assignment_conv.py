import numpy as np
import importlib
# import matplotlib.pyplot as plt
from src.B_Model_fitting import assignment_multilogit as am
importlib.reload(am)

def fuel_consumptions(fuel_efficiencies, utilisation_profiles):
    return None

def conv_assignt(conn_data, existence_cache, alphas, betas, omegas_p, ranges, constraints, indices, epsilon = 1e-4):
    omegas = omegas_p.copy()
    diffs =np.array(1)*existence_cache[indices,0]
    weights = conn_data[:, 3][np.newaxis, :]
    mask = existence_cache[indices, 0]
    # L_diffs = []
    while np.abs(diffs).max() > epsilon:
        speed_adapt = 1.4
        if np.abs(diffs).max() > 0.2 :
            speed_adapt = 0.95
        elif np.abs(diffs).max() > 0.1 :
            speed_adapt = 1.05
        elif np.abs(diffs).max() > 0.05 :
            speed_adapt = 1.2
        pots = am.predict_assignt(conn_data, existence_cache, alphas, betas, omegas, ranges, d_norm =1, cap_norm = 1, single_p = True)
        ass_mod = pots[indices, :] * weights
        ass_const = ass_mod.sum(axis=1)
        diffs = (np.log(constraints) - np.log(ass_const))*mask
        omegas[indices]+= speed_adapt*diffs[:, np.newaxis]
        # L_diffs.append(diffs)
    # plt.plot(np.array(L_diffs))
    # plt.ylim(-0.05,0.05)
    # plt.show()
    return omegas

def range_distance(pots, conn_data, ranges):
    ass_mod = pots * conn_data[:, 3][np.newaxis, :]
    avg_rg = ((ass_mod*conn_data[:, 1][np.newaxis, :]).sum(axis=1))/ass_mod.sum(axis=1)
    return avg_rg


