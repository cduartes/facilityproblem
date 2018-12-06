import numpy as np
from typing import List
import math
import argparse
import itertools, time

parser = argparse.ArgumentParser(
    description='FacilityProblem solver with Tabu Search.'
)
parser.add_argument('file', help='Name of the file to load')
args = parser.parse_args()

class Tabu(object):
    """ Tabu Class
    Must work on vectors?
    ------
    Parameters

    """
    
    def __init__(self):
        print("something")

    def tabu_search(self, s0):
        s_prime = s0
        g_prime2 = g_func(s0)

        if( is_feasible(s0)):
            s_prime = s0
            g_prime1 = g_func(s0)
        else:
            s_prime = None
            g_prime1 = float('Inf')
        #tabu(j,l) = -1
        alpha = 1
        return 1

    def is_tabu(self):
        return False

    def g_func2(self, s):
        return 1


def readFile(fileName: str):
    f_fixed = []
    f_capacities = []
    c_costs = []
    c_demand = []
    with open(fileName, "r") as f:
        nLocations, nClients = f.readline().strip().split()
        nLocations = int(nLocations)
        nClients = int(nClients)
        for i in range(nLocations):
            b_i, f_i = f.readline().strip().split()
            f_fixed.append(float(f_i))
            f_capacities.append(float(b_i))
        for i  in range(nClients):
            demand = int(f.readline().strip())
            c_demand.append(demand)
            lines = math.ceil(nLocations / 7)
            clientsCost = []
            for i in range(lines):
                line = f.readline().strip().split()
                clientsCost += line
            clientsCost = np.array([float(x) for x in clientsCost])
            c_costs.append(clientsCost)
    f_fixed = np.array(f_fixed)
    f_capacities = np.array(f_capacities)
    c_costs = np.array(c_costs)
    c_demand = np.array(c_demand)
    return f_fixed, f_capacities, c_costs, c_demand

def initialSolution(f_fixed, f_capacities, c_costs, c_demand):
    def getRatio(elem):
        return elem['ratio']
    order = []
    demand_total = c_demand.sum()
    for index, data in enumerate(zip(f_fixed, f_capacities)):
        ratio = data[0]/data[1]
        order.append({ 'index': index, 'ratio': ratio})
    order = sorted(order, key=getRatio)
    X = np.zeros((f_fixed.shape[0], c_demand.shape[0]))
    for o in order:
        X[o['index']][c_costs[o['index']].argmin()] = 1
        demand_total -= c_demand[c_costs[o['index']].argmin()]
    return X

def is_feasible(s):
    return True

def is_satisfying(s):
    return True

def perturbation(s):
    return s

def g_func(s):
    return 1

def show_result(s):
    print("results")

if __name__ == "__main__":
    f_fixed, f_capacities, c_costs, c_demand = readFile(args.file)
    demand_total = c_demand.sum()
    s_0 = initialSolution(f_fixed, f_capacities, c_costs, c_demand)

    # (f_capacities, f_fixed_costs, c_demands, suplying_costs, s) = init()
    
    # tabu = np.negative(np.ones((1,3)))
    # print(tabu)

    # # the solution is represented by a vector
    # tabu = Tabu()
    # s0 = initialSolution(s)
    # k = 3
    # s_hat = tabu.tabu_search(s0)
    # if (is_feasible(s_hat)):
    #     s_it = s_hat
    #     g_it = g_func(s_it)
    # else:
    #     #s_it = None
    #     g_it = float('Inf')
    
    # for v in range(k):
    #     s_prim = perturbation(s_it)
    #     s_sl = tabu.tabu_search(s_prim)
    #     if(is_feasible(s_sl) and g_func(s_sl) < g_it):
    #         s_it = s_sl
    #         g_it = g_func(s_sl)
    #         omega = 0
    #     if(is_feasible(s_sl) and is_satisfying(s_sl)):
    #         s_hat = s_sl

    # if(is_satisfying(s_hat)):
    #     show_result(s_hat)