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
        sum2 = 0
        for j in range(0, c_demand.shape[0]):
            sum2 += c_demand[j] * s.X[i][j]
        sum2 -= f_capacities[i] * s.Y[i]
        sum1 += max(0, sum2)
    if sum1 == 0:
        return True
    else:
        return False

def show_result(s, g, c_demand, f_capacities):
    print("results")
    print("Demand vs Capacity per facility")
    for index, facility in enumerate(s.X):
        suma = 0
        clients = np.argwhere(facility == 1)
        for client in clients:
            suma += c_demand[client[0]]
        print("facility {}: {}/{} - {}".format(index, suma, f_capacities[index], len(clients)))
    print(s.X)
    print(s.Y)
    print(g)

def is_satisfying(s, delta, g_prime):
    if g(s) < g_prime * (1 + delta):
        return True
    return False

def option1(s, c_costs):
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
                    cost_min = c_f[client]
                    cost_index = index
        s.X[cost_index, client] = 1
    return s

def option3(s, f_capacities, c_demand):
    rows = []
    for index, row in enumerate(s.X):
        if row.sum() > 0:
            rows.append(index)
    if rows:
        selected = np.random.choice(rows, 1)[0]
        posibles = []
        clients = np.argwhere(s.X[selected] == 1)
        total_demand = 0
        for cord in clients:
            total_demand += c_demand[cord[0]]
        for index, facility in enumerate(f_capacities):
            if index != selected and facility >= total_demand:
                posibles.append(index)
        if posibles:
            transfer_facility = np.random.choice(posibles, 1)[0]
            s.X[transfer_facility] += s.X[selected]
            s.X[selected] = np.zeros(s.X[selected].shape)
    return s

def option4(s):
    facilities = np.argwhere(s.Y == 1)
    print('facilities open: ',len(facilities))
    clients = np.argwhere(s.X == 1)
    for client in clients:
        if facilities:
            selected = np.random.choice(facilities, 1)[0]
            s.X[client[0], client[1]] = 0
            s.X[selected, client[1]] = 1
    return s



def perturbation(s, c_costs, f_capacities, c_demand):
    option = np.random.randint(1, 4)
    if option == 1:
        print('\npertubation: 1')
        s_ = option1(s, c_costs)
    if option == 2:
        print('\npertubation: 3')
        s_ = option3(s, f_capacities, c_demand)
    if option == 3:
        print('\npertubation: 4')
        s_ = option4(s)
    return s_

def g(s0): #TODO: PREGUNTAR A MATIAS POR alpha*q
    sum2 = 0
    for index in range(len(s0.fixed)):
        sum2 += s0.Y[index] * s0.fixed[index]
    sum1 = 0
    for i, c_i in enumerate(s0.costs):
        for j, c_j  in enumerate(c_i):
            sum1 += c_j * s0.X[i][j]
    return sum1 + sum2

def q(s, c_demand, f_capacities):
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
    return sum1

if __name__ == "__main__":
    '''
    s_prime = s'
    s_ast = s*
    g_prime = g*
    s_dev = S'    
    S_ = S-
    '''
    K = 10
    delta = 0.05
    f_fixed, f_capacities, c_costs, c_demand = readFile(args.file)
    X, Y = initialSolution(f_fixed, f_capacities, c_costs, c_demand)
    s0 = Solution(X, Y, c_costs, f_fixed)
    tabu = Tabu(g, q, is_feasible, X.shape, K)
    s_hat = tabu.tabu_search(s0.copy(), c_demand, f_capacities)
    if (is_feasible(s_hat, c_demand, f_capacities)):
        s_ast = s_hat.copy()
        g_ast = g(s_ast)
    else:
        s_prime = None
        g_ast = float('inf')
    for i in tqdm(range(K)):
        s_prime = perturbation(s_hat.copy(), c_costs, f_capacities, c_demand)
        S_ = tabu.tabu_search(s_prime.copy(), c_demand, f_capacities)
        if is_feasible(S_, c_demand, f_capacities) and g(S_) < g_ast:
            s_ast = S_.copy()
            g_ast = g(S_)
            theta = 0
        #  TODO: if S_ feasible and S_ satisfies the acceptance criteriion set s_hat = S_
        if is_feasible(S_, c_demand, f_capacities) and is_satisfying(S_, delta, g_ast):
            s_hat = S_.copy()
    show_result(s_ast, g_ast, c_demand, f_capacities)
    
    