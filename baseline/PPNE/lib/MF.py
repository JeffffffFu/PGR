import torch.nn as nn
import torch
import scipy.sparse as sparse
import numpy as np
from pdb import set_trace as st

class DeepWalkMF(nn.Module):
    def __init__(self,args, T=10, b=5):
        """
        `vol_G` is the sum of the adjacency matrix A.
        `d_rt` is the an array containing the degree of each node.
        `T` is the window size used in skip-gram
        `b` is the number of negative examples.
        """
        super(DeepWalkMF, self).__init__()
        self.T = T
        self.b = b # Number of negative samples.
        self.args = args

    def set_d_rt(self, d_rt):
        self.d_rt = d_rt
        D_rt_inv = sparse.diags(self.d_rt ** -1)
        self.D_inv = torch.tensor(D_rt_inv.toarray()).to(self.args.device)
        # self.D_inv = torch.tensor(D_rt_inv.toarray())
        self.vol_G = self.d_rt.sum()

    def forward(self, X):
        """
        X: adjacency matrix
        D: diagonal matrix where each element in the diagonal represents a degree
        T: window size in DeepWalk
        """
        X.requires_grad_(True)

        self.S = torch.zeros_like(X).to(self.args.device)
        # self.S = torch.zeros_like(X)
        X = torch.mm(self.D_inv, X)
        X_power = torch.eye(X.shape[0], dtype=torch.float64).to(self.args.device)
        # X_power = torch.eye(X.shape[0], dtype=torch.float64)
        for i in range(self.T):
            X_power = torch.mm(X, X_power)
            self.S += X_power
        output = self.S * (self.vol_G / self.T / self.b)
        output = torch.mm(output, self.D_inv)
        output = torch.clamp(output, 0.001)
        output = torch.log(output)

        return output


# class DeepWalkMF_float(nn.Module):
#     def __init__(self, T=10, b=5):
#         """
#         `vol_G` is the sum of the adjacency matrix A.
#         `d_rt` is the an array containing the degree of each node.
#         `T` is the window size used in skip-gram
#         `b` is the number of negative examples.
#         """
#         super(DeepWalkMF_float, self).__init__()
#         self.T = T
#         self.b = b # Number of negative samples.

#     def set_d_rt(self, d_rt):
#         self.d_rt = d_rt
#         D_rt_inv = sparse.diags(self.d_rt ** -1)
#         self.D_inv = torch.tensor(D_rt_inv.toarray()).cuda()
#         # self.D_inv = torch.tensor(D_rt_inv.toarray())
#         self.vol_G = self.d_rt.sum()

#     def forward(self, X):
#         """
#         X: adjacency matrix
#         D: diagonal matrix where each element in the diagonal represents a degree
#         T: window size in DeepWalk
#         """
#         X.requires_grad_(True)

#         self.S = torch.zeros_like(X).cuda()
#         # self.S = torch.zeros_like(X)
#         X = torch.mm(self.D_inv, X)
#         X_power = torch.eye(X.shape[0], dtype=torch.float32).cuda()
#         # X_power = torch.eye(X.shape[0], dtype=torch.float32)
#         for i in range(self.T):
#             X_power = torch.mm(X, X_power)
#             self.S += X_power
#         output = self.S * (self.vol_G / self.T / self.b)
#         output = torch.mm(output, self.D_inv)
#         output = torch.clamp(output, 0.001)
#         output = torch.log(output)

#         return output


# class DeepWalkMF_v2(nn.Module):
#     def __init__(self, T=10, b=5):
#         """
#         `vol_G` is the sum of the adjacency matrix A.
#         `d_rt` is the an array containing the degree of each node.
#         `T` is the window size used in skip-gram
#         `b` is the number of negative examples.
#         """
#         super(DeepWalkMF_v2, self).__init__()
#         self.T = T
#         self.b = b # Number of negative samples.

#     def set_d_rt(self, d_rt):
#         self.d_rt = d_rt
#         D_rt_inv = sparse.diags(self.d_rt ** -1)
#         self.D_inv = torch.tensor(D_rt_inv.toarray()).cuda()
#         # self.D_inv = torch.tensor(D_rt_inv.toarray())
#         self.vol_G = self.d_rt.sum()

#     def forward(self, X):
#         """
#         X: adjacency matrix
#         D: diagonal matrix where each element in the diagonal represents a degree
#         T: window size in DeepWalk
#         """
#         d_rt =torch.sum(X, 0)
#         d_rt = d_rt ** -1
#         X.requires_grad_(True)

#         #           self.S = torch.zeros_like(X).cuda()
#         X = X * torch.transpose(d_rt[:,None], dim0=1,dim1=0)
#         #           X_power = torch.eye(X.shape[0], dtype=torch.float64).cuda()
#         X_power = X
#         self.S = X
#         for i in range(self.T-1):
#             X_power = torch.mm(X, X_power)
#             self.S = torch.add(self.S, X_power)
#         output = self.S * (self.vol_G / self.T / self.b)
#         # output = torch.mm(output, self.D_inv)
#         output = output * d_rt[:,None]
#         # if self.netmf:
#         X = torch.clamp(X, 0.001)
#         output = torch.log(output)
#         # output = X

#         return output

class LINEMF(nn.Module):
    def __init__(self,args, b=5):
        """
        `vol_G` is the sum of the adjacency matrix A.
        `d_rt` is the an array containing the degree of each node.
        `T` is the window size used in skip-gram
        `b` is the number of negative examples.
        """
        super(LINEMF, self).__init__()
        self.b = b # Number of negative samples.

    def set_d_rt(self, d_rt):
        self.d_rt = d_rt
        D_rt_inv = sparse.diags(d_rt ** -1)
        # D_rt_inv = np.diag(d_rt ** -1)
        self.D_inv = torch.tensor(D_rt_inv.toarray()).to(self.args.device)
        # self.D_inv = torch.tensor(D_rt_inv.toarray())
        # self.D_inv = torch.tensor(D_rt_inv.toarray())
        #           self.D_inv = torch.tensor(D_rt_inv, dtype=torch.float32).cuda()
        self.vol_G = d_rt.sum()

    def forward(self, X):
        """
        X: adjacency matrix
        D: diagonal matrix where each element in the diagonal represents a degree
        T: window size in DeepWalk
        """
        X.requires_grad_(True)

        X = X * (self.vol_G / self.b)
        X = torch.mm(X, self.D_inv)
        X = torch.mm(self.D_inv, X)
        X = torch.clamp(X, 0.001)
        output = torch.log(X)
        return output


# class LINEMF_cpu(nn.Module):
#     def __init__(self, b=5):
#         """
#         `vol_G` is the sum of the adjacency matrix A.
#         `d_rt` is the an array containing the degree of each node.
#         `T` is the window size used in skip-gram
#         `b` is the number of negative examples.
#         """
#         super(LINEMF_cpu, self).__init__()
#         self.b = b # Number of negative samples.

#     def set_d_rt(self, d_rt):
#         self.d_rt = d_rt
#         D_rt_inv = sparse.diags(d_rt ** -1)
#         # D_rt_inv = np.diag(d_rt ** -1)
#         self.D_inv = torch.tensor(D_rt_inv.toarray()).cuda()
#         # self.D_inv = torch.tensor(D_rt_inv.toarray())
#         # self.D_inv = torch.tensor(D_rt_inv, dtype=torch.float32).cuda()
#         self.vol_G = d_rt.sum()

#     def forward(self, X):
#         """
#         X: adjacency matrix
#         D: diagonal matrix where each element in the diagonal represents a degree
#         T: window size in DeepWalk
#         """
#         X.requires_grad_(True)

#         X = X * (self.vol_G / self.b)

#         X = torch.mm(X, self.D_inv)
#         X = torch.mm(self.D_inv, X)
#         X = torch.clamp(X, 0.001)
#         output = torch.log(X)
#         return output


# class LINEMF_fac(nn.Module):
#     def __init__(self, b=5):
#         """
#         `vol_G` is the sum of the adjacency matrix A.
#         `d_rt` is the an array containing the degree of each node.
#         `T` is the window size used in skip-gram
#         `b` is the number of negative examples.
#         """
#         super(LINEMF_fac, self).__init__()
#         self.b = b  # Number of negative samples.

#     def set_d_rt(self, d_rt):
#         self.d_rt = d_rt
#         D_rt_inv = sparse.diags(d_rt ** -1)
#         # D_rt_inv = np.diag(d_rt ** -1)
#         self.D_inv = torch.tensor(D_rt_inv.toarray()).cuda()
#         # self.D_inv = torch.tensor(D_rt_inv.toarray())
#         # self.D_inv = torch.tensor(D_rt_inv, dtype=torch.float32).cuda()
#         self.vol_G = d_rt.sum()

#     def forward(self, X):
#         """
#         X: adjacency matrix
#         D: diagonal matrix where each element in the diagonal represents a degree
#         T: window size in DeepWalk
#         """
#         X.requires_grad_(True)

#         X = X * (self.vol_G / self.b)

#         X_res = torch.zeros(size=(50000, 50000)).cpu()
#         for i in range(50):
#             y_part = self.D_inv[:, i * 1000:(i + 1) * 1000].cuda()
#             # y_part = self.D_inv[:, i * 1000:(i + 1) * 1000]
#             X_res[:, i * 1000:(i + 1) * 1000] = torch.mm(X, y_part)

#         # 
#         X = X_res

#         X_res = torch.zeros(size=(50000, 50000)).cpu()
#         for i in range(50):
#             y_part = self.D_inv[i * 1000:(i + 1) * 1000, :].cuda()
#             # y_part = self.D_inv[i * 1000:(i + 1) * 1000, :]
#             X_res[i * 1000:(i + 1) * 1000, :] = torch.mm(y_part, X)
#         # X = torch.mm(X, self.D_inv)
#         # X = torch.mm(self.D_inv, X)
#         # X = X_res.cuda()
#         X = X_res

#         X = torch.clamp(X, 0.001)
#         output = torch.log(X)
#         return output
