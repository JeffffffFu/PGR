import numba
import numpy as np
import scipy.linalg as spl
import scipy.sparse.linalg as sspl


@numba.jit(nopython=True)
def estimate_loss_with_delta_eigenvals(candidates, flip_indicator, vals_org, vecs_org, n_nodes, dim, window_size):
    """Computes the estimated loss using the change in the eigenvalues for every candidate edge flip.

    :param candidates: np.ndarray, shape [?, 2]
        Candidate set of edge flips,
    :param flip_indicator: np.ndarray, shape [?]
        Vector indicating whether we are adding an edge (+1) or removing an edge (-1)
    :param vals_org: np.ndarray, shape [n]
        The generalized eigenvalues of the clean graph
    :param vecs_org: np.ndarray, shape [n, n]
        The generalized eigenvectors of the clean graph
    :param n_nodes: int
        Number of nodes
    :param dim: int
        Embedding dimension
    :param window_size: int
        Size of the window
    :return: np.ndarray, shape [?]
        Estimated loss for each candidate flip
    """

    loss_est = np.zeros(len(candidates))
    for x in range(len(candidates)):
        i, j = candidates[x]
        vals_est = vals_org + flip_indicator[x] * (
                2 * vecs_org[i] * vecs_org[j] - vals_org * (vecs_org[i] ** 2 + vecs_org[j] ** 2))

        vals_sum_powers = sum_of_powers(vals_est, window_size)

        loss_ij = np.sqrt(np.sum(np.sort(vals_sum_powers ** 2)[:n_nodes - dim]))
        loss_est[x] = loss_ij

    return loss_est


def perturbation_top_flips(adj_matrix, candidates, n_flips, dim, window_size):
    """Selects the bottom (n_flips) number of flips using our perturbation attack.

    :param adj_matrix: sp.spmatrix
        The graph represented as a sparse scipy matrix
    :param candidates: np.ndarray, shape [?, 2]
        Candidate set of edge flips
    :param n_flips: int
        Number of flips to select
    :param dim: int
        Dimensionality of the embeddings.
    :param window_size: int
        Co-occurence window size.
    :return: np.ndarray, shape [?, 2]
        The bottom edge flips from the candidate set
    """
    n_nodes = adj_matrix.shape[0]
    # vector indicating whether we are adding an edge (+1) or removing an edge (-1)
    delta_w = 1 - 2 * adj_matrix[candidates[:, 0], candidates[:, 1]].A1

    # generalized eigenvalues/eigenvectors
    deg_matrix = np.diag(adj_matrix.sum(1).A1)
    vals_org, vecs_org = spl.eigh(adj_matrix.toarray(), deg_matrix)

    loss_for_candidates = estimate_loss_with_delta_eigenvals(candidates, delta_w, vals_org, vecs_org, n_nodes,
                                                             dim, window_size)
    top_flips = candidates[loss_for_candidates.argsort()[-n_flips:]]

    return top_flips


def perturbation_bottom_flips(adj_matrix, candidates, n_flips, dim, window_size):
    """Selects the bottom (n_flips) number of flips using our perturbation attack.

    :param adj_matrix: sp.spmatrix
        The graph represented as a sparse scipy matrix
    :param candidates: np.ndarray, shape [?, 2]
        Candidate set of edge flips
    :param n_flips: int
        Number of flips to select
    :param dim: int
        Dimensionality of the embeddings.
    :param window_size: int
        Co-occurence window size.
    :return: np.ndarray, shape [?, 2]
        The bottom edge flips from the candidate set
    """
    n_nodes = adj_matrix.shape[0]
    # vector indicating whether we are adding an edge (+1) or removing an edge (-1)
    delta_w = 1 - 2 * adj_matrix[candidates[:, 0], candidates[:, 1]].A1

    # generalized eigenvalues/eigenvectors
    deg_matrix = np.diag(adj_matrix.sum(1).A1)
    vals_org, vecs_org = spl.eigh(adj_matrix.toarray(), deg_matrix)

    loss_for_candidates = estimate_loss_with_delta_eigenvals(candidates, delta_w, vals_org, vecs_org, n_nodes,
                                                             dim, window_size)
    bottom_flips = candidates[loss_for_candidates.argsort()[:n_flips]]

    return bottom_flips


@numba.jit(nopython=True)
def sum_of_powers(x, power):
    """For each x_i, computes \sum_{r=1}^{pow) x_i^r (elementwise sum of powers).

    :param x: shape [?]
        Any vector
    :param pow: int
        The largest power to consider
    :return: shape [?]
        Vector where each element is the sum of powers from 1 to pow.
    """
    n = x.shape[0]
    sum_powers = np.zeros((power, n))
    for i, i_power in enumerate(range(1, power + 1)):
        sum_powers[i] = np.power(x, i_power)

    return sum_powers.sum(0)


def perturbation_utility_loss(adj_matrix, candidates, dim, window_size):
    """Selects the bottom (n_flips) number of flips using our perturbation attack.

    :param adj_matrix: sp.spmatrix
        The graph represented as a sparse scipy matrix
    :param candidates: np.ndarray, shape [?, 2]
        Candidate set of edge flips
    :param n_flips: int
        Number of flips to select
    :param dim: int
        Dimensionality of the embeddings.
    :param window_size: int
        Co-occurence window size.
    :return: np.ndarray, shape [?, 2]
        The bottom edge flips from the candidate set
    """
    n_nodes = adj_matrix.shape[0]
    # vector indicating whether we are adding an edge (+1) or removing an edge (-1)
    delta_w = 1 - 2 * adj_matrix[candidates[:, 0], candidates[:, 1]].A1

    # generalized eigenvalues/eigenvectors
    deg_matrix = np.diag(adj_matrix.sum(1).A1)
    vals_org, vecs_org = spl.eigh(adj_matrix.toarray(), deg_matrix)

    loss_for_candidates = estimate_loss_with_delta_eigenvals(candidates, delta_w, vals_org, vecs_org, n_nodes,
                                                             dim, window_size)
    # bottom_flips = candidates[loss_for_candidates.argsort()[:n_flips]]

    # return bottom_flips

    return loss_for_candidates


def original_utility_loss(adj_matrix, dim, window_size):
    """Selects the bottom (n_flips) number of flips using our perturbation attack.

    :param adj_matrix: sp.spmatrix
        The graph represented as a sparse scipy matrix
    :param candidates: np.ndarray, shape [?, 2]
        Candidate set of edge flips
    :param n_flips: int
        Number of flips to select
    :param dim: int
        Dimensionality of the embeddings.
    :param window_size: int
        Co-occurence window size.
    :return: np.ndarray, shape [?, 2]
        The bottom edge flips from the candidate set
    """
    n_nodes = adj_matrix.shape[0]
    # vector indicating whether we are adding an edge (+1) or removing an edge (-1)

    # generalized eigenvalues/eigenvectors
    deg_matrix = np.diag(adj_matrix.sum(1).A1)
    vals_org, vecs_org = spl.eigh(adj_matrix.toarray(), deg_matrix)

    # loss_for_candidates = estimate_loss_with_delta_eigenvals(candidates, delta_w, vals_org, vecs_org, n_nodes,
    #                                                          dim, window_size)
    # bottom_flips = candidates[loss_for_candidates.argsort()[:n_flips]]
    vals_sum_powers = sum_of_powers(vals_org, window_size)
    loss = np.sqrt(np.sum(np.sort(vals_sum_powers ** 2)[:n_nodes - dim]))

    return loss

