#!/user/bin/python
# author jeff
import math

from privacy_analyze.RDP.compute_rdp import compute_rdp
from privacy_analyze.RDP.rdp_convert_dp import compute_eps


# from absl import app



def compute_dp_sgd_privacy(n, batch_size, noise_multiplier, epochs, delta):

    q = batch_size / n  # q - the sampling ratio.
    if q > 1:
        print ('n must be larger than the batch size.')
    orders = (list(range(2, 64)) + [128, 256, 512])

    steps = int(math.ceil(epochs * (n / batch_size)))

    return apply_dp_sgd_analysis(q, noise_multiplier, steps, orders, delta)

def apply_dp_sgd_analysis(q, sigma, steps, orders, delta):
  """Compute and print results of DP-SGD analysis."""

  # compute_rdp requires that sigma be the ratio of the standard deviation of
  # the Gaussian noise to the l2-sensitivity of the function to which it is
  # added. Hence, sigma here corresponds to the `noise_multiplier` parameter   sigma=noise_multilpier
  # in the DP-SGD implementation found in privacy.optimizers.dp_optimizer
  rdp = compute_rdp(q, sigma, steps, orders)

  eps, opt_order = compute_eps(orders, rdp, delta)




  return eps, opt_order


def apply_dp_sgd_analysis_old(q, sigma, steps, orders, delta):
  """Compute and print results of DP-SGD analysis."""

  # compute_rdp requires that sigma be the ratio of the standard deviation of
  # the Gaussian noise to the l2-sensitivity of the function to which it is
  # added. Hence, sigma here corresponds to the `noise_multiplier` parameter   sigma=noise_multilpier
  # in the DP-SGD implementation found in privacy.optimizers.dp_optimizer
  rdp = compute_rdp(q, sigma, steps, orders)

  eps, opt_order = compute_eps(orders, rdp, delta)



  return eps, opt_order

'''
orders = (list(range(2, 64)) + [128, 256, 512])  # 默认的lamda
eps, opt_order=apply_dp_sgd_analysis(256/60000, 1.1, 17470, orders, 10**(-5))
print("eps:",format(eps)+"| order:",format(opt_order))
'''

#
if __name__=="__main__":
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64))+ [128, 256, 512]

    eps, opt_order = apply_dp_sgd_analysis(0.1, 2.0, 5, orders, 10 ** (-5))
    print("eps:", format(eps) + "| order:", format(opt_order))