import numpy as np


def delivery_scenario(range_m, shares_p, vol_p, vol_fleet):
    print('faire une fonction qui permet de stabiliser l indicateur de range, tout en  lissant les volumes')
    print('integrer aussi la variable d age de retrait')
    print('ou bien tracer 5 hypothèses caricaturales, et ventiler les livraisons entre les scénarios')
    return None



def fleet_content(a, b, fleet_obs_t, retirement_coeffs, constraint, epsilon=0.0001, first = False):
    if first :
        fleets_b = b**(np.exp(-retirement_coeffs)) * fleet_obs_t
        if fleets_b.sum().sum() < constraint :
            print('temporary aircraft parking', end=' ')
            # return fleet_content(b, 1,fleet_obs_t, retirement_coeffs, constraint, epsilon)
            # print(b)
            return fleet_content(b, b*2, fleet_obs_t, retirement_coeffs, constraint, epsilon, first = True)

    fleets = ((a+b)/2)**(np.exp(-retirement_coeffs)) * fleet_obs_t
    e = (fleets.sum().sum() - constraint)/constraint
    # print('power: '+str((a+b)/2) + ' constraint: '+str(constraint) + ' '+str(fleets.sum().sum())+' error: '+str(np.abs(e)))
    if b==0:
        print('reso bug')
    if np.abs(e) < epsilon:
        return fleets, (a+b)/2
    elif e > 0:
        return fleet_content(a, (a+b)/2,fleet_obs_t, retirement_coeffs, constraint, epsilon)
    else:
        return fleet_content((a+b)/2, b,fleet_obs_t, retirement_coeffs, constraint, epsilon)


def retirement_content():
    fleet_n_1 = None
    fleet_n_0 = None
    r =  fleet_n_1-fleet_n_0
    return r