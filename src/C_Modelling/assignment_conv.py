import numpy as np
import importlib
from src.B_Model_fitting import assignment_multilogit as am
importlib.reload(am)

def fuel_consumptions(fuel_efficiencies, utilisation_profiles):
    return None

def conv_assignt(conn_data, existence_cache, alphas, betas, omegas_p, ranges, constraints, indices, speed = 1.01,  epsilon = 0.0001):
    omegas = omegas_p.copy()
    #loop
    diffs =np.array(1)
    while np.abs(diffs).max() > epsilon:
        pots = am.predict_assignt(conn_data, existence_cache, alphas, betas, omegas, ranges, d_norm =1, cap_norm = 1)
        ass_mod = pots[indices, :] * conn_data[:, 3][np.newaxis, :]
        ass_const = ass_mod.sum(axis=1)
        diffs = np.log(constraints) - np.log(ass_const[indices])
        omegas[indices]+= speed*diffs[:, np.newaxis]

    return omegas

def range_distance(pots, conn_data, ranges):
    ass_mod = pots * conn_data[:, 3][np.newaxis, :]
    avg_rg = ((ass_mod*conn_data[:, 1][np.newaxis, :]).sum(axis=1))/ass_mod.sum(axis=1)
    return avg_rg


