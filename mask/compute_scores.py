import math

import numpy as np
import torch
from math import ceil
from typing import Callable, Iterable, List, Optional, Union
import random

def keep_edges_add_many_edge_from_zero_priD2(meta_grads, last_mask, device, indexs,edge_num_gen,
                                                     origin_matrix_vec_non_zero_indices,matrix_vec,pri_d):
    meta_grads = meta_grads.to(device)
    last_mask = last_mask.to(device)

    scores = [torch.zeros_like(mask) for mask in meta_grads]
    scores = [curr + g for curr, g in zip(scores, meta_grads)]
    scores_vec = torch.cat([score.flatten() for score in scores])
    torch.cuda.empty_cache()

    edge_matrix = [torch.zeros_like(mask) for mask in last_mask]
    edge_matrix = [curr + g for curr, g in zip(edge_matrix, last_mask)]
    edge_matrix_vec = torch.cat([score.flatten() for score in edge_matrix])
    last_mask_matrix_vec_non_zero_indices=torch.nonzero(edge_matrix_vec)

    torch.cuda.empty_cache()

    edge_matrix2 = [torch.zeros_like(mask) for mask in last_mask]
    edge_matrix2 = [curr + g for curr, g in zip(edge_matrix2, last_mask)]
    edge_matrix_vec2 = torch.cat([score.flatten() for score in edge_matrix2])
    torch.cuda.empty_cache()

#    min_values, topk_indices = torch.min(scores_vec, dim=0)
 #   print("min_values:", min_values)
    edge_matrix_vec[indexs] = 1


    edge_matrix_vec[origin_matrix_vec_non_zero_indices] = 1

    nonzero_indices = torch.nonzero(edge_matrix_vec).squeeze()
    scores_vec[nonzero_indices] = 100.
    min_values, topk_indices = torch.min(scores_vec, dim=0)
    print("min_values:", min_values)
    edge_matrix_vec2[topk_indices] = 1.0


    keep_masks = edge_matrix_vec2.view(meta_grads.shape)
    return keep_masks
