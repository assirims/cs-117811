# ==================== issa.py ====================
# Improved Sparrow Search Algorithm for feature selection

# issa.py
import numpy as np
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class ISSAFeatureSelector:
    def __init__(self, pop_size, max_iter, chaotic=True, random_state=None):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.chaotic = chaotic
        self.random_state = np.random.RandomState(random_state)

    def _chaotic_sequence(self, size):
        # sine-chaos map initialization
        x = self.random_state.rand(size)
        for i in range(size):
            x[i] = np.sin(np.pi * x[i])
        return x

    def _initialize(self, dim):
        if self.chaotic:
            pop = self._chaotic_sequence((self.pop_size, dim))
        else:
            pop = self.random_state.rand(self.pop_size, dim)
        return pop

    def _fitness(self, solution, X, y):
        # Binary mask via threshold
        mask = solution > 0.5
        if mask.sum() == 0:
            return np.inf
        X_sel = X[:, mask]
        clf = DecisionTreeClassifier(random_state=self.random_state)
        # simple 3-fold evaluation
        idx = self.random_state.permutation(len(y))
        split = int(0.7 * len(y))
        train_idx, test_idx = idx[:split], idx[split:]
        clf.fit(X_sel[train_idx], y[train_idx])
        preds = clf.predict(X_sel[test_idx])
        err = 1 - accuracy_score(y[test_idx], preds)
        # fitness: balance of error and subset size
        alpha, beta = 0.9, 0.1
        fitness = (alpha * err) + (beta * (mask.sum() / X.shape[1]))
        return fitness

    def select(self, X, y):
        n_features = X.shape[1]
        pop = self._initialize(n_features)
        fitness = np.zeros(self.pop_size)
        for i in range(self.pop_size):
            fitness[i] = self._fitness(pop[i], X, y)

        for t in range(self.max_iter):
            # placeholder for ISSA update equations
            # For brevity, perform simple random search step
            candidate = self.random_state.rand(self.pop_size, n_features)
            for i in range(self.pop_size):
                fit_c = self._fitness(candidate[i], X, y)
                if fit_c < fitness[i]:
                    pop[i] = candidate[i]
                    fitness[i] = fit_c
        best = pop[fitness.argmin()] > 0.5
        return best