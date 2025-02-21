import numpy as np
import torch
import torch_geometric
from torch_geometric.datasets import Planetoid



def compare_adj(A,B):

    edge_A=torch.nonzero(A).size(0)-len(A[0])
    edge_B=torch.nonzero(B).size(0)-len(A[0])


    A[A != 0] = 1
    B[B != 0] = 1

    C=A-B
    ones_mask = C.eq(1.0)
    negative_ones_mask=C.eq(-1.0)
    count_ones = ones_mask.sum().item()
    count_negative_ones=negative_ones_mask.sum().item()
    L1_norm = C.abs().sum()

   # print("count_negative_ones:",count_negative_ones)
    print("diff:",count_negative_ones/edge_A)
    diff=count_negative_ones/edge_A
    return diff

def compare_adj2(A,B):

    C=B-A
    diff = torch.count_nonzero(C).item()
    return diff


def compare_adj3(A,B):

    C=B+A
    count = torch.sum(torch.eq(C, 2))

    return count

if __name__=='__main__':


    list=[0.4,0.4001,0.4002,0.4003,0.4004]
    t=len(list)
    for i in range(t):
        for j in range(i,t):

            File_Path1=f'C://python flie/GNN_DP/result3/regen_diff/Cora()/{list[i]}/101'
            A=torch.load(f"{File_Path1}/edge.pth")


            File_Path2=f'C://python flie/GNN_DP/result3/regen_diff/Cora()/{list[j]}/101'
            B=torch.load(f"{File_Path2}/edge.pth")

            diff=compare_adj(A,B)