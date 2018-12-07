import numpy as np

class Tabu(object):
    """ Tabu Class
    Must work on vectors?
    ------
    Parameters

    """
    def __init__(self, g, feasible, n, gamma):
        self.g_func = g
        self.is_feasible = feasible
        self.matrix = np.negative(np.ones(n))
        self.alpha = 1
        self.gamma = gamma
        

    def tabu_search(self, s0, c_demand, f_capacities):
        self.s_prime = s0.X
        self.g_prime2 = self.g_func(s0)

        if (self.is_feasible(s0, c_demand, f_capacities)):
            self.g_prime1 = self.g_func(s0)
        else:
            self.s_prime = None
            self.g_prime1 = float('Inf')
        #tabu(j,l) = -1
        self.s = s0.X
        self.generate_neighbours(self.s)
        # for x in range(gamma):

        return 1

    def generate_neighbours(self, matrix):
        dict_of_neighbours = {}
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                if(matrix[i][j]!=0):
                    if j not in dict_of_neighbours:
                        _list = list()
                        _list.append([i,matrix[i][j]])
                        dict_of_neighbours[j]=_list
                    else:
                        dict_of_neighbours[j].append([i,matrix[i][j]])
                    if i not in dict_of_neighbours:
                        _list = list()
                        _list.append([j,matrix[i][j]])
                        dict_of_neighbours[i]=_list
                    else:
                        dict_of_neighbours[i].append([i,matrix[i][j]])
        return dict_of_neighbours

    def is_tabu(self):
        return False

    def g_func2(self, s):
        return 1