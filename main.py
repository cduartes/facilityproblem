import numpy as np
from typing import List
import math
import argparse
import itertools, time
from TabuSearch import Tabu

parser = argparse.ArgumentParser(
    description='FacilityProblem solver with Tabu Search.'
)
parser.add_argument('file', help='Name of the file to load')
args = parser.parse_args()

class Solution():
    def __init__(self, X, Y, costs, fixed):
        self.X = X
        self.Y = Y
        self.costs = costs
        self.fixed = fixed

def readFile(fileName: str):
    f_fixed = []
    f_capacities = []
    c_costs = []
    c_demand = []
    with open(fileName, "r") as f:
        nLocations, nClients = f.readline().strip().split()
        nLocations = int(nLocations)
        nClients = int(nClients)
        for _ in range(nLocations):
            b_i, f_i = f.readline().strip().split()
            f_fixed.append(float(f_i))
            f_capacities.append(float(b_i))
        for _ in range(nClients):
            demand = int(f.readline().strip())
            c_demand.append(demand)
            lines = math.ceil(nLocations / 7)
            clientsCost = []
            for _ in range(lines):
                line = f.readline().strip().split()
                clientsCost += line
            clientsCost = np.array([float(x) for x in clientsCost])
            c_costs.append(clientsCost)
    f_fixed = np.array(f_fixed)
    f_capacities = np.array(f_capacities)
    c_costs = np.array(c_costs).transpose()
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
    Y = np.zeros(f_fixed.shape[0])
    list_selected = []
    copy_costs = c_costs.copy()
    for o in order:
        Y[o['index']] = 1
        demand_total -= f_capacities[o['index']]
        capacity = f_capacities[o['index']]
        while capacity > 0 and len(list_selected) < 1000:
            index = copy_costs[o['index']].argmin()
            copy_costs[o['index']][index] = float('Inf')
            if (index not in list_selected):
                list_selected.append(index)
                X[o['index']][index] = 1
                capacity -= c_demand[index]
        if (demand_total <= 0):
            break
    print((Y*f_capacities).sum())
    print(len(list_selected))
    return X, Y

def is_feasible(s):
    return True

def is_satisfying(s):
    return True

def perturbation(s):
    return s

def g(s0):
    sum2 = 0
    for index in range(len(s0.fixed)):
        sum2 += s0.Y[index] * s0.fixed[index]
    sum1 = 0
    for i, c_i in enumerate(s0.costs):
        for j, c_j  in enumerate(c_i):
            sum1 += c_j * s0.X[i][j]
    return sum1 + sum2

def show_result(s):
    print("results")

if __name__ == "__main__":
    f_fixed, f_capacities, c_costs, c_demand = readFile(args.file)
    demand_total = c_demand.sum()
    X, Y = initialSolution(f_fixed, f_capacities, c_costs, c_demand)
    s0 = Solution(X, Y, c_costs, f_fixed)
    tabu = Tabu(g, is_feasible, X.shape, 100)
    tabu.tabu_search(s0)
    g(s0)

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