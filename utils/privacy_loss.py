import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import torch


def privacy_loss_RR(p,q):
    priL=0.
    if p >= 0.5:
        TP = p * q
        FP = (1 - p)*(1 - q)
        priL=TP/(TP+FP)
    else:
        TP=(1-p)*q
        FP=p*(1-q)
        priL=TP/(TP+FP)
    return priL

if __name__=="__main__":
#    q=10858/(2708*2707)
    q=9746/(3312*3311)
    p=0.99
    priL = privacy_loss_RR(p, q)
    print(priL)

# if __name__=="__main__":
#
#
#     p_list=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.9,0.9,1.0]
#     priL_liss=[]
#     q=10858/(2708*2707)
#     for p in p_list:
#         priL=privacy_loss_RR(p, q)
#         priL_liss.append(priL)
#     plt.plot(p_list, priL_liss,marker="o")
#     plt.xlabel('p',fontsize=16)
#     plt.ylabel('priL',fontsize=16)
#
#     plt.show()
