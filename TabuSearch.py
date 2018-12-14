import numpy as np
from Solution import Solution
from tqdm import tqdm

class Tabu(object):
    """ Tabu Class
    Must work on vectors?
    ------
    Parameters

    """
    def __init__(self, g, q, feasible, n, gamma, beta=0.75, epsilon=0.2):
        self.g_func = g
        self.q_func = q
        self.is_feasible = feasible
        self.matrix = np.negative(np.ones(n))
        self.alpha = 1
        self.gamma = gamma
        self.beta = beta
        self.min_g = float('inf')
        self.epsilon = epsilon
        
    def tabu_search(self, s0, c_demand, f_capacities):
        '''
        s_prime = S~*
        g_prime1 = g1*
        g_prime2 = g2*
        s_prime_hat = S!(hat)*
        S = S
        '''
        self.s_prime = s0.copy()
        self.g_prime2 = self.g_func(s0.copy())
        self.c_demand = c_demand
        self.f_capacities = f_capacities
        if (self.is_feasible(s0, c_demand, f_capacities)):
            self.s_prime_hat= s0.copy()
            self.g_prime1 = self.g_func(s0.copy())
        else:
            self.s_prime_hat= None
            self.g_prime1 = float('Inf')
        self.S = s0.copy()
        for _ in range(self.gamma):
            # Decrease Tabu count
            tabus = np.argwhere(self.matrix >= 0)
            for cord in tabus:
                self.matrix[cord[0], cord[1]] -= 1
            # Generate Neighbors and Evaluate them
            # (S_, g_sol: best solution and g(s_), selected: movement, top: list of best solutions )
            N1, N2 = self.generate_neighbours(self.S.X)
            S_, g_sol, selected, top = self.evaluate_neighbors(self.S, N1, N2)
            if not (self.is_feasible(S_, c_demand, f_capacities)):
                print(len(top))
                for s_hat in top[1:]:
                    if (self.is_feasible(s_hat[0], c_demand, f_capacities) and s_hat[1] < self.g_prime1):
                        self.s_prime_hat= s_hat[0].copy()  #  solucion
                        self.g_prime1 = s_hat[1]  #  evaluacion
                        break
            if g_sol < self.g_prime2:
                self.s_prime = S_.copy()
                self.g_prime2 = g_sol 
            if (self.is_feasible(S_, c_demand, f_capacities) and g_sol < self.g_prime1):
                self.s_prime_hat= S_.copy()
                self.g_prime1 = g_sol
            if len(selected) == 3:
                self.matrix[selected[2], selected[0]] = 7 # OMEGA = 7 + m(a/u); a: numero de veces que este movimiento se ha realizado; u: valor máximo de movimientos (a).
            else:
                self.matrix[selected[2], selected[1]] = 7
                self.matrix[selected[3], selected[0]] = 7
            #  TODO: paso 11 algoritmo
            self.S = self.s_prime.copy()
            self.update_alpha(self.q_func(self.S, c_demand, f_capacities))
        if self.s_prime_hat:
            return self.s_prime_hat.copy()
        else:
            return self.s_prime.copy()


    def generate_neighbours(self, matrix):
        '''
        generate neighbors
        [(0, 1, 4)]
        [(0, 4, 1, 4)]
        '''
        #print(self.matrix)
        clients = matrix.copy().transpose()
        neighbors = []
        for i, c in enumerate(clients):
            original_facility = np.where(c == 1)[0][0]
            for x in range(matrix.shape[0]):
                if np.random.random() <= self.beta:
                    #  TODO: self.matrix[x, 1] no debe evaluarse aca. falta criterio de aspiracion
                    #if x != original_facility and self.matrix[x, i] < 0:
                    if x != original_facility:
                        neighbors.append((i, original_facility, x))
        neighbors2 = []
        for i, c in enumerate(clients):
            original_facility = np.where(c == 1)[0][0]
            for j, k in enumerate(clients):
                if j == i:
                    continue
                if np.random.random() <= self.beta:
                    original_facility2 = np.where(k == 1)[0][0]
                    #if original_facility2 != original_facility and self.matrix[original_facility2, i] < 0:
                    if original_facility2 != original_facility:
                        neighbors2.append((i, j, original_facility, original_facility2))
        return neighbors, neighbors2

    def evaluate_neighbors(self, solution, N1, N2):
        '''
        Evaluate Neighbors
        sol is the best solution
        g_sol is the evaluation
        selected is the movement (client, origin, destination)
        top is a list of the best movements: [(selected, g )]
        '''
        min_g = float('inf')
        selected = None
        S_ = None
         #(is_aspirated(self.matrix[x, i]) or not(is_tabu(self.matrix[x, i]))
        top = []

        
        # penalizations = np.argwhere(self.matrix >= 0)
        # print(penalizations)

        # for f, c in enumerate(penalizations):
        #     if 

        for pos in N1:
            testing = solution.X.copy()
            testing[pos[1], pos[0]] = 0
            testing[pos[2], pos[0]] = 1
            Y = [1 if x.sum() > 0 else 0 for x in testing]
            test_sol = Solution(testing.copy(), Y, solution.costs, solution.fixed)
            sol = self.g_func(test_sol)
            flag = self.is_feasible(test_sol, self.c_demand, self.f_capacities)
            if sol < min_g and (self.is_aspirated(sol, flag) or self.matrix[pos[2], pos[0]] < 0):
                min_g = sol
                selected = pos
                S_ = test_sol
                top.insert(0, (test_sol.copy(), min_g, selected))
        for pos in N2:
            testing = solution.X.copy()
            testing[pos[2], pos[0]] = 0
            testing[pos[3], pos[1]] = 0
            testing[pos[2], pos[1]] = 1
            testing[pos[3], pos[0]] = 1
            Y = [1 if x.sum() > 0 else 0 for x in testing]
            test_sol = Solution(testing.copy(), Y, solution.costs, solution.fixed)
            sol = self.g_func(test_sol)
            flag = self.is_feasible(test_sol, self.c_demand, self.f_capacities)
            if sol < min_g  and (self.is_aspirated(sol, flag) or self.matrix[pos[2], pos[1]] < 0):
                min_g = sol
                selected = pos
                S_= test_sol
                top.insert(0, (test_sol.copy(), min_g, selected))
        self.min_g = sol
        return S_, sol, selected, top

    def is_aspirated(self, eval, flag):
        if flag and eval < self.g_prime1:
            return True
        elif eval <  self.g_prime2:
            return True
        else:
            return False

    def update_alpha(self, q_s):
        if q_s > 0:
            self.alpha = self.alpha / 1 + self.epsilon
        else:
            self.alpha = self.alpha * (1 + self.epsilon)
        