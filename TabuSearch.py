import numpy as np
from Solution import Solution
from tqdm import tqdm

class Tabu(object):
    """ Tabu Class
    Must work on vectors?
    ------
    Parameters

    """
    def __init__(self, g, feasible, n, gamma, beta=0.75, epsilon=0.1):
        self.g_func = g
        self.is_feasible = feasible
        self.matrix = np.negative(np.ones(n))
        self.alpha = 1
        self.gamma = gamma
        self.beta = beta
        self.min_g = float('inf')
        self.epsilon = epsilon
        

    def tabu_search(self, s0, c_demand, f_capacities):
        self.s_prime = s0
        self.g_prime2 = self.g_func(s0)

        if (self.is_feasible(s0, c_demand, f_capacities)):
            self.g_prime1 = self.g_func(s0)
            self.s_prime_hat= s0
        else:
            self.s_prime_hat= None
            self.g_prime1 = float('Inf')
        self.S = s0
        for _ in range(self.gamma):
            # Decrease Tabu count
            tabus = np.argwhere(self.matrix >= 0)
            for cord in tabus:
                self.matrix[cord[0], cord[1]] -= 1
            # Generate Neighbors and Evaluate them
            # (S_, g_sol: best solution and g(s), selected: movement, top: list of best solutions )
            N1, N2 = self.generate_neighbours(self.S.X)
            S_, g_sol, selected, top = self.evaluate_neighbors(self.S, N1, N2)
            if not (self.is_feasible(S_, c_demand, f_capacities)):
                for t in top:
                    if (self.is_feasible(t[0], c_demand, f_capacities) and t[1] < self.g_prime1):
                        self.s_prime_hat= t[0]
                        self.g_prime1 = t[1]
                        break
            if g_sol < self.g_prime2:
                self.s_prime = S_
                self.g_prime2 = g_sol 
            if (self.is_feasible(S_, c_demand, f_capacities) and g_sol < self.g_prime1):
                self.s_prime_hat= S_
                self.g_prime1 = g_sol
            if len(selected) == 3:
                self.matrix[selected[2], selected[0]] = 7
            else:
                self.matrix[selected[2], selected[1]] = 7
                self.matrix[selected[3], selected[0]] = 7
            self.S = self.s_prime
        if self.s_prime_hat:
            return self.s_prime_hat
        else:
            return self.s_prime

    def generate_neighbours(self, matrix):
        clients = matrix.copy().transpose()
        neighbors = []
        for i, c in enumerate(clients):
            original_facility = np.where(c == 1)[0][0]
            for x in range(matrix.shape[0]):
                if np.random.random() <= self.beta:
                    if x != original_facility and self.matrix[x, i] < 0:
                        neighbors.append((i, original_facility, x))
        neighbors2 = []
        for i, c in enumerate(clients):
            original_facility = np.where(c == 1)[0][0]
            for j, k in enumerate(clients):
                if j == i:
                    continue
                if np.random.random() <= self.beta:
                    original_facility2 = np.where(k == 1)[0][0]
                    if original_facility2 != original_facility and self.matrix[original_facility2, i] < 0:
                        neighbors2.append((i, j, original_facility, original_facility2))
        return neighbors, neighbors2

    def evaluate_neighbors(self, solution, N1, N2):
        min_g = float('inf')
        selected = None
        S_ = None
        top = []
        for pos in N1:
            testing = solution.X.copy()
            testing[pos[1], pos[0]] = 0
            testing[pos[2], pos[0]] = 1
            Y = [1 if x.sum() > 0 else 0 for x in testing]
            test_sol = Solution(testing, Y, solution.costs, solution.fixed)
            sol = self.g_func(test_sol)
            if sol < min_g:
                top.insert(0, (selected, min_g, selected))
                min_g = sol
                selected = pos
                S_ = test_sol
        for pos in N2:
            testing = solution.X.copy()
            testing[pos[2], pos[0]] = 0
            testing[pos[3], pos[1]] = 0
            testing[pos[2], pos[1]] = 1
            testing[pos[3], pos[0]] = 1
            Y = [1 if x.sum() > 0 else 0 for x in testing]
            test_sol = Solution(testing, Y, solution.costs, solution.fixed)
            sol = self.g_func(test_sol)
            if sol < min_g:
                top.insert(0, (selected, min_g, selected))
                min_g = sol
                selected = pos
                S_= test_sol
        self.min_g = sol
        return S_, sol, selected, top

    def is_tabu(self):
        return False

    def g_func2(self, s):
        return 1