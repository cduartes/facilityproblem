class Tabu(object):
    """ Tabu Class
    Must work on vectors?
    ------
    Parameters


    """
    
    def __init__(self, gamma):
        gamma
        print("something")

    def search(self, s0):
        s_prime = s0
        g_prime2 = g_func(s0)

        if( is_feasible(s0)):
            s_prime = s0
            g_prime1 = g_func(s0)
        else:
            s_prime = None
            g_prime1 = float('Inf')
        tabu(j,l) = -1
        alpha = 1

        for p in range(gamma):


        return 1

    def is_tabu(self):
        return False

    def g_func2(self, s):
        return 1

def initialSolution():
    return 1

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
    # the solution is represented by a vector
    tabu = Tabu()
    s0 = initialSolution()
    k = 3
    s_hat = tabu.search(s0)
    if (is_feasible(s_hat)):
        s_it = s_hat
        g_it = g_func(s_it)
    else:
        #s_it = None
        g_it = float('Inf')
    
    for v in range(k):
        s_prim = perturbation(s_it)
        s_sl = tabu.search(s_prim)
        if(is_feasible(s_sl) and g_func(s_sl) < g_it):
            s_it = s_sl
            g_it = g_func(s_sl)
            omega = 0
        if(is_feasible(s_sl) and is_satisfying(s_sl)):
            s_hat = s_sl

    if(is_satisfying(s_hat)):
        show_result(s_hat)