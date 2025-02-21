import math
from scipy.stats import laplace
import numpy as np
import matplotlib.pyplot as plt


def probability_X1_greater_X0(eps):
    x=1/eps
    return 1- (((2*x + 1)/(4 * x)) * np.exp(-1/x))

def probability_of_group(n, k):
    numerator = math.factorial(n) // math.factorial(n - k)
    denominator = math.factorial(k)
    fector=(n/k)**k
    return fector * denominator / numerator




if __name__ == "__main__":
    n=100
    k=5
    q=probability_of_group(n, k)
    print("q:",q)
    eps = np.arange(5,6,0.5)
    #eps = 2
    p=probability_X1_greater_X0(eps)

    p1=p**(k*(n-k))

    p2=(p**(n-k))*q

    plt.figure(figsize=(10, 6))
    plt.plot(eps, p1, label='P(X1 > X0)', color='b')
    plt.plot(eps, p2, label='P(X1 > X0)-group', color='r')

    plt.xlabel('eps')
    plt.ylabel('Probability')
    plt.title('Probability of X1 > X0 with Laplace Noise')
    plt.legend()
    plt.grid(True)
    #plt.axhline(y=0.5, color='r', linestyle='--')  # 添加参考线 y=0.5
    plt.show()
