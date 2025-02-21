import numpy as np
from pdb import set_trace as st


def generate_mask(A, num_per_row=10):
    # For each row, we randomly pick `num_per_row` cells to modify.
    n = A.shape[0]
    A = np.zeros((n,n))
    for i in range(n):
        mask = np.random.choice(n, num_per_row)
        A[i][mask] = 1
    mask = (A == 1)
    return mask


def generate_mask_from_0(A, num_per_row=10):
    n = A.shape[0]
    mask0 = (A == 0)
    A_return = np.zeros_like(A)
    for i in range(n):
        index_list = np.argwhere(mask0[i]).T[0]
        mask = np.random.choice(index_list, num_per_row, replace=False)
        A_return[i][mask] = 1
    mask = (A_return == 1)
    return mask


def project_add_delete_edge_with_mask(A_adv, A_org, num_add, num_delete, mask, verbose=True, return_edge=False):
    """
    We can only add or delete edges which is not in the mask.
    `True` in `mask` means that we can modify the the corresponding cells.
    """
    mask0 = (A_org == 0)
    mask1 = (A_org == 1)
    A_return = np.copy(A_org)

    mask0 = mask0 & mask
    mask1 = mask1 & mask

    loc0 = (A_adv - A_org)
    loc0[~mask0] = -100
    index_sort = np.argsort(np.ravel(loc0))[::-1]
    if verbose:
        print("Following edges are added")
    for i in index_sort[:num_add]:
        index = np.unravel_index(i, dims=loc0.shape)
        if verbose:
            print("Location %s value %f"%(index, loc0[index]))
        A_return[index] = 1
        A_return.T[index] = 1

    loc1 = (A_org - A_adv)
    loc1[~mask1] = -100
    index_sort = np.argsort(np.ravel(loc1))[::-1]
    if verbose:
        print("Following edges are removed")
    for i in index_sort[:num_delete]:
        index = np.unravel_index(i, dims=loc1.shape)
        if verbose:
            print("Location %13s value %f "%(index, loc1[index]))
        A_return[index] = 0
        A_return.T[index] = 0

    if not return_edge:
        return A_return
    else:
        return (A_return, index)


def project_add_delete_total_with_mask(A_adv, A_org, num_change, mask, verbose=True):
    """
    We can only add or delete edges which is not in the mask.
    `True` in `mask` means that we can modify the the correponding cells.
    """
    mask0 = (A_org == 0)
    mask1 = (A_org == 1)
    A_return = np.copy(A_org)

    mask0 = mask0 & mask
    mask1 = mask1 & mask

    loc0 = (A_adv - A_org)
    loc0[~mask0] = -1000

    loc1 = (A_org - A_adv)
    loc1[~mask1] = -1000

    loc_total = np.maximum(loc0, loc1)
    index_sort = np.argsort(np.ravel(loc_total))[::-1]
    if verbose:
        print("Following edges are changed")
    for i in index_sort[:num_change]:
        index = np.unravel_index(i, dims=loc0.shape)
        if A_org[index] == 0:
            A_return[index] = 1
            A_return.T[index] = 1
            if verbose:
                print("Add location %s value %f"%(index, loc_total[index]))
        else:
            A_return[index] = 0
            A_return.T[index] = 0
            if verbose:
                print("Delete location %s value %f"%(index, loc_total[index]))

    return A_return


def random_add_delete_edge_with_mask_v2(A, num_add, num_delete, mask, verbose=True):
    mask1 = (A==0)
    mask0 = ~mask1
    mask1 = mask1 & mask
    A = np.copy(A)
    nonzero_indexes = mask1.nonzero()
    chosen_indexes = np.random.choice(nonzero_indexes[0].shape[0], int(num_add), replace=False)
    nonzero_x = nonzero_indexes[0][chosen_indexes]
    nonzero_y = nonzero_indexes[1][chosen_indexes]
    if verbose:
        print("Following edges are added:")
    for i in zip(nonzero_x, nonzero_y):
        if verbose:
            print("Edges %s"%(i,))
    # assert nonzero_x > nonzero_y, "In random_add_delete_edge_with_mask function"
    A[nonzero_x, nonzero_y] = 1
    # A[nonzero_y, nonzero_x] = 1
 
    # mask0 = mask0 & mask
    # nonzero_indexes = mask0.nonzero()
    # chosen_indexes = np.random.choice(nonzero_indexes[0].shape[0], int(num_delete), replace=False)
    # nonzero_x = nonzero_indexes[0][chosen_indexes]
    # nonzero_y = nonzero_indexes[1][chosen_indexes]
    # A[nonzero_x, nonzero_y] = 0
    return A


def random_add_delete_edge_with_mask(A, num_add, num_delete, mask, verbose=True):
    mask1 = (A==0)
    mask0 = ~mask1
    mask1 = mask1 & mask
    A = np.copy(A)
    nonzero_indexes = mask1.nonzero()
    chosen_indexes = np.random.choice(nonzero_indexes[0].shape[0], int(num_add), replace=False)
    nonzero_x = nonzero_indexes[0][chosen_indexes]
    nonzero_y = nonzero_indexes[1][chosen_indexes]
    if verbose:
        print("Following edges are added:")
    for i in zip(nonzero_x, nonzero_y):
        if verbose:
            print("Edges %s"%(i,))
    # assert nonzero_x > nonzero_y, "In random_add_delete_edge_with_mask function"
    A[nonzero_x, nonzero_y] = 1
    A[nonzero_y, nonzero_x] = 1
 
    # mask0 = mask0 & mask
    # nonzero_indexes = mask0.nonzero()
    # chosen_indexes = np.random.choice(nonzero_indexes[0].shape[0], int(num_delete), replace=False)
    # nonzero_x = nonzero_indexes[0][chosen_indexes]
    # nonzero_y = nonzero_indexes[1][chosen_indexes]
    # A[nonzero_x, nonzero_y] = 0
    return A


def random_delete_edge_with_mask(A, num_delete, mask, verbose=True):
    mask1 = (A==0)
    mask0 = ~mask1
    A = np.copy(A)
 
    mask0 = mask0 & mask
    nonzero_indexes = mask0.nonzero()
    chosen_indexes = np.random.choice(nonzero_indexes[0].shape[0], int(num_delete), replace=False)
    nonzero_x = nonzero_indexes[0][chosen_indexes]
    nonzero_y = nonzero_indexes[1][chosen_indexes]
    if verbose:
        print("Following edges are deleted:")
    for i in zip(nonzero_x, nonzero_y):
        if verbose:
            print("Edges %s"%(i,))
    A[nonzero_x, nonzero_y] = 0
    A[nonzero_y, nonzero_x] = 0
    return A