import numpy as np
import scipy.spatial
import scipy.optimize


def error(A, Ahat):
    # mean square error, choosing the minimal permution with linear sum assignment
    C = scipy.spatial.distance.cdist(A.T, Ahat.T) ** 2
    D = scipy.optimize.linear_sum_assignment(C)
    return C[D[0], D[1]].sum() / (A.shape[0] * A.shape[1])


def errorSAD(A, Ahat):
    table = (A.T @ Ahat)
    normA = np.linalg.norm(A, axis=0)
    normAhat = np.linalg.norm(Ahat, axis=0)
    table = np.arccos(table / (np.outer(normA, normAhat)))
    D = scipy.optimize.linear_sum_assignment(table)
    return table[D[0], D[1]].mean(), D[1], table[D[0], D[1]]
