import numpy as np

from pdb import set_trace as st


def project_01(A):
    """
    Project inplace the matrix A to the space of 0,1
    """
    mask = (A > 1)
    A[mask] = 1
    mask = (A < 0)
    A[mask] = 0
    return


def project_add_delete_edge(A_adv, A_org, num_add, num_delete, verbose=True):
    # A_adv, A_org should both be numpy array
    mask0 = (A_org == 0)
    mask1 = (A_org == 1)
    A_return = np.copy(A_org)

    loc0 = (A_adv - A_org)
    loc0[mask1] = -100
    index_sort = np.argsort(np.ravel(loc0))[::-1]
    if verbose:
        print("Following edges are added")
    for i in index_sort[:num_add]:
        index = np.unravel_index(i, dims=loc0.shape)
        if verbose:
            print(index)
        A_return[index] = 1

    loc1 = (A_org - A_adv)
    loc1[mask0] = -100
    index_sort = np.argsort(np.ravel(loc1))[::-1]
    # print("Following edges are removed")
    for i in index_sort[:num_delete]:
        index = np.unravel_index(i, dims=loc1.shape)
        # print(index)
        A_return[index] = 0
    return A_return


def random_add_delete_edge(A, num_add, num_delete):
    mask0 = (A == 0)
    mask1 = (A==1)
    A = np.copy(A)
    nonzero_indexes = mask0.nonzero()
    chosen_indexes = np.random.choice(nonzero_indexes[0].shape[0], int(num_add))
    nonzero_x = nonzero_indexes[0][chosen_indexes]
    nonzero_y = nonzero_indexes[1][chosen_indexes]
    A[nonzero_x, nonzero_y] = 1

    nonzero_indexes = mask1.nonzero()
    chosen_indexes = np.random.choice(nonzero_indexes[0].shape[0], int(num_delete))
    nonzero_x = nonzero_indexes[0][chosen_indexes]
    nonzero_y = nonzero_indexes[1][chosen_indexes]
    A[nonzero_x, nonzero_y] = 0
    return A