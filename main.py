import numpy as np
from typing import List
import math
import argparse
import itertools, time
from TabuSearch import Tabu
from Solution import Solution
from tqdm import tqdm

parser = argparse.ArgumentParser(
    description='FacilityProblem solver with Tabu Search.'
)
parser.add_argument('file', help='Name of the file to load')
args = parser.parse_args()

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
        if capacity == 0:
            continue
        while capacity > 0 and len(list_selected) < c_costs.shape[1]:
            index = copy_costs[o['index']].argmin()
            copy_costs[o['index']][index] = float('Inf')
            if (index not in list_selected):
                list_selected.append(index)
                X[o['index']][index] = 1
                capacity -= c_demand[index]
        if (demand_total <= 0):
            break
    return X, Y

def is_feasible(s, c_demand, f_capacities):
    """
    NOTE: Solutions are allowed to be infeasible during the neighborhood search
    A solution is infeasible if it violates the capacity constraint of the facilities.
    J = d.shape[0] is the number of customers
    I = b.shape[0] is the number of facilities
    """
    sum1 = 0
    sum2 = 0

    for i in range(0, f_capacities.shape[0]):
        for j in range(0, c_demand.shape[0]):
            sum2 = c_demand[j]*s.X[i][j] - f_capacities[i]*s.Y[i]
        sum1 += max(0, sum2)
    if sum1 == 0:
        return True
    else:
        return False

def is_satisfying(s):
    return True

def option1(s):
    rows = []
    for index, row in enumerate(s.X):
        if row.sum() == 1:
            rows.append(index)
    if rows:
        selected = np.random.choice(rows, 1)[0]
        client = s.X[selected].argmax()
        s.X[selected, client] = 0
        s.Y = [1 if x.sum() > 0 else 0 for x in s.X]
        cost_min = float('inf')
        cost_index = None
        for index, c_f in enumerate(c_costs):
            if s.Y[index] == 1 and index != selected:
                if cost_min > c_f[client]:
                    c_min = c_f[client]
                    cost_index = index
        s.X[cost_index, client] = 1
    return s

def option2(s):
    pass

def perturbation(s, c_costs):
    option = np.random.randint(1, 6)
    if True:
        s_ = option1(s)
    return s_

def g(s0):
    sum2 = 0
    for index in range(len(s0.fixed)):
        sum2 += s0.Y[index] * s0.fixed[index]
    sum1 = 0
    for i, c_i in enumerate(s0.costs):
        for j, c_j  in enumerate(c_i):
            sum1 += c_j * s0.X[i][j]
    return sum1 + sum2

def show_result(s, g):
    print("results")
    print(s.X)
    print(s.Y)
    print(g)
    

if __name__ == "__main__":
    K = 10
    f_fixed, f_capacities, c_costs, c_demand = readFile(args.file)
    demand_total = c_demand.sum()
    X, Y = initialSolution(f_fixed, f_capacities, c_costs, c_demand)
    s0 = Solution(X, Y, c_costs, f_fixed)
    tabu = Tabu(g, is_feasible, X.shape, 10)
    s_hat = tabu.tabu_search(s0, c_demand, f_capacities)
    if (is_feasible(s_hat, c_demand, f_capacities)):
        s_prime = s_hat
        g_prime = g(s_hat)
    else:
        s_prime = None
        g_prime = float('inf')
    for i in tqdm(range(K)):
        s_dev = perturbation(s_hat, c_costs)
        S_ = tabu.tabu_search(s_dev, c_demand, f_capacities)
        if is_feasible(S_, c_demand, f_capacities) and g(s_prime) < g_prime:
            s_prime = S_
            g_prime = g(s_prime)
            theta = 0
    show_result(s_prime, g_prime)
    
    