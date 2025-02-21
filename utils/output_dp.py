import math
import random

import numpy as np
import torch
def output_dp_gaussian(output,C,sigma,device):
    new_out_put=[]
    output=output.to('cpu')
    output=output.detach().numpy()
    for vector in output:
        norm = np.linalg.norm(vector)
        if norm > C:
            vector=vector*(C/norm)
        vector_noise=vector+np.random.normal(0,sigma*C,len(vector))

        new_out_put.append(vector_noise)
    return torch.tensor(new_out_put,device=device)

def output_dp_lap(output,C,eps,device):
    new_out_put=[]
    output=output.to('cpu')
    output=output.detach().numpy()
    for vector in output:
        norm = np.linalg.norm(vector, ord=1)
        if norm > C:
            vector=vector*(C/norm)
        vector_noise=vector+np.random.laplace(0,C/eps,len(vector))

        new_out_put.append(vector_noise)
    return torch.tensor(new_out_put,device=device)

def output_dp_KRR(output,label_max,eps,device):
    new_out_put=[]
    output=output.max(1)[1]
    output=output.to('cpu')
    output=output.detach().numpy()
    p=math.exp(eps)/(math.exp(eps)+label_max-1)  #
    for vector in output:
        if random.random() > p:
            KRR_list = [i for i in range(label_max) if i != vector]

            vector_RR= random.choice(KRR_list)
            new_out_put.append(vector_RR)
        else:
            new_out_put.append(vector)
    return torch.tensor(new_out_put,device=device)