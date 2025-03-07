
import scipy.stats
import numpy as np

from GCN_DP_PyG.privacy_alysis.RDP.compute_rdp import _compute_rdp
from GCN_DP_PyG.privacy_alysis.RDP.rdp_convert_dp import compute_eps


def compute_multiterm_rdp(orders, num_training_steps, noise_multiplier, num_samples, max_terms_per_node, batch_size):
    terms_rv = scipy.stats.hypergeom(num_samples, max_terms_per_node, batch_size)
    terms_logprobs = [
        terms_rv.logpmf(i) for i in np.arange(max_terms_per_node + 1)
    ]

    # Compute unamplified RDPs (that is, with sampling probability = 1).
    rdp = np.array([_compute_rdp(1.0, noise_multiplier, order) for order in
                    orders])
    unamplified_rdps = rdp

    # Compute amplified RDPs for each (order, unamplified RDP) pair.
    amplified_rdps = []
    for order, unamplified_rdp in zip(orders, unamplified_rdps):
        beta = unamplified_rdp * (order - 1)
        log_fs = beta * (
            np.square(np.arange(max_terms_per_node + 1) / max_terms_per_node))
        amplified_rdp = scipy.special.logsumexp(terms_logprobs + log_fs) / (
                order - 1)
        amplified_rdps.append(amplified_rdp)

    # Verify lower bound.
    amplified_rdps = np.asarray(amplified_rdps)
    if not np.all(unamplified_rdps *
                  (batch_size / num_samples) ** 2 <= amplified_rdps + 1e-6):
        raise ValueError('The lower bound has been violated. Something is wrong.')

    # Account for multiple training steps.
    amplified_rdps_total = amplified_rdps * num_training_steps
    return amplified_rdps_total

if __name__ == "__main__":
    # orders = (list(range(2, 64)) + [128, 256, 512])  # 默认的lamda
    orders = np.arange(1, 10, 0.1)[1:]
    num_training_steps = 10
    noise_multiplier = 1
    target_delta = 1e-5
    batch_size = 10
    num_samples = 1000
    max_terms_per_node = 1
    rdp_every_epoch = compute_multiterm_rdp(orders, num_training_steps, noise_multiplier, num_samples,
                                            max_terms_per_node, batch_size)
    epsilon, best_alpha = compute_eps(orders, rdp_every_epoch, target_delta)
    print(f"multiterm_epsilon:{epsilon}")