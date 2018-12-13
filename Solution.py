class Solution():
    def __init__(self, X, Y, costs, fixed):
        self.X = X
        self.Y = Y
        self.costs = costs
        self.fixed = fixed

    def copy(self):
        return type(self)(
            self.X.copy(),
            self.Y.copy(),
            self.costs.copy(),
            self.fixed.copy()
        )