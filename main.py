import numpy as np
import itertools, time

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

def init():
    f_capacities = [] 
    f_fixed_costs = []
    with open("capa8000", "r") as f:
        n, m = f.readline()[1:-2].split(' ')
        print(n)
        s = np.zeros(int(m)) # representation of the solution, all facilities closed
        print(np.shape(s))
        costs = []
        for i in range(0, int(n)):
            b_i, f_i = f.readline()[1:-2].split(' ')
            f_capacities.append(b_i)
            f_fixed_costs.append(f_i)
        
        f_capacities = [float(x) for x in f_capacities]
        f_fixed_costs = [float(x) for x in f_fixed_costs]

        c_demands = []
        print(m)
        for i in range(0, int(m)):
            c_demands.append(f.readline()[1:-2])
            print(c_demands)

            for c in range(0, 14):
                line = f.readline()[1:-2].split(' ')
                print(line)
        
        time.sleep(5.5)    # pause 5.5 seconds

        
        c_demands = [float(x) for x in c_demands]
        
        for i in range(1, int(n)):
            cost_for_i_to_j = f.readline()[:-2].split(' ')
            cost_for_i_to_j = [float(x) for x in cost_for_i_to_j]
            costs.append(cost_for_i_to_j)
        suplying_costs = np.array(costs)
    return f_capacities, f_fixed_costs, c_demands, suplying_costs, s

def initialSolution(s):
    return s

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
    (f_capacities, f_fixed_costs, c_demands, suplying_costs, s) = init()
    
    tabu = np.negative(np.ones((1,3)))
    print(tabu)

    # the solution is represented by a vector
    tabu = Tabu()
    s0 = initialSolution(s)
    k = 3
    s_hat = tabu.tabu_search(s0)
    if (is_feasible(s_hat)):
        s_it = s_hat
        g_it = g_func(s_it)
    else:
        #s_it = None
        g_it = float('Inf')
    
    for v in range(k):
        s_prim = perturbation(s_it)
        s_sl = tabu.tabu_search(s_prim)
        if(is_feasible(s_sl) and g_func(s_sl) < g_it):
            s_it = s_sl
            g_it = g_func(s_sl)
            omega = 0
        if(is_feasible(s_sl) and is_satisfying(s_sl)):
            s_hat = s_sl

    if(is_satisfying(s_hat)):
        show_result(s_hat)