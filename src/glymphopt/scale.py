import numpy as np


class ReducedProblem:
    def __init__(self, problem, S, x0):
        self.problem = problem
        self.S = np.asarray(S)
        self.x0 = np.asarray(x0)

    def transform(self, y):
        return self.S.dot(y) + self.x0

    def F(self, y):
        return self.problem.F(self.S.dot(y) + self.x0)

    def gradF(self, y):
        x = self.S.dot(y) + self.x0
        gradF = self.problem.gradF(x)
        return self.S.T @ gradF

    def hessp(self, y, dy):
        x = self.S.dot(y) + self.x0
        dx = self.S.dot(dy)
        return self.S.T @ self.problem.hessp(x, dx)

    def hess(self, y):
        return np.array([self.hessp(y, ei) for ei in np.eye(len(y))])


def create_reduced_problem(problem, x_expected, indices):
    n = len(x_expected)
    eye = np.eye(n)
    S = np.array([x_expected * eye[i] for i in indices]).T
    offset = np.array([x_expected * eye[j] for j in range(n) if j not in indices]).sum(
        axis=0
    )
    return ReducedProblem(problem, S, offset)
