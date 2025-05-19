# ==================== roa.py ====================
# Rabbit Optimization Algorithm for hyperparameter tuning

# roa.py
import numpy as np

class ROAOptimizer:
    def __init__(self, pop_size, max_iter, param_ranges, random_state=None):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.param_ranges = param_ranges
        self.rs = np.random.RandomState(random_state)

    def _initialize(self):
        # positions: each solution is [lr, batch_size]
        sol = []
        for _ in range(self.pop_size):
            lr = self.rs.uniform(*self.param_ranges['learning_rate'])
            bs = self.rs.randint(*self.param_ranges['batch_size'])
            sol.append([lr, bs])
        return np.array(sol)

    def optimize(self, train_fn):
        # train_fn: function accepting (lr, batch) and returning validation accuracy
        sols = self._initialize()
        fitness = np.zeros(self.pop_size)
        for i, s in enumerate(sols):
            fitness[i] = 1 - train_fn(lr=s[0], batch_size=int(s[1]))

        for t in range(self.max_iter):
            # placeholder for ROA update: simple random perturbation
            candidate = sols + self.rs.normal(scale=0.1, size=sols.shape)
            # enforce bounds
            candidate[:,0] = np.clip(candidate[:,0], *self.param_ranges['learning_rate'])
            candidate[:,1] = np.clip(candidate[:,1], *self.param_ranges['batch_size'])
            for i in range(self.pop_size):
                fit_c = 1 - train_fn(lr=candidate[i,0], batch_size=int(candidate[i,1]))
                if fit_c < fitness[i]:
                    sols[i] = candidate[i]
                    fitness[i] = fit_c
        best_idx = fitness.argmin()
        return {'learning_rate': sols[best_idx,0], 'batch_size': int(sols[best_idx,1])}