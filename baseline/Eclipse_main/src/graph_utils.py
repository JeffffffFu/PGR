import multiprocessing
import operator
import numpy as np


def getCommunityCounts(adjacency, labels, rng, dp=False, eps=2.0):
    comm_edges = {}
    comm_edges_nondp = {}
    comm_dict = {}
    for i in range(len(adjacency)):
        if not labels[i] in comm_dict:
            comm_dict[labels[i]] = []
        comm_dict[labels[i]].append(i)
    comms = [val for key, val in sorted(comm_dict.items(), key=operator.itemgetter(0))]

    for i in range(len(adjacency)):
        friends = set(np.where(adjacency[i] > 0)[0])
        comm_edges_nondp[i] = [len(friends.intersection(set(x))) for x in comms]
        comm_edges[i] = comm_edges_nondp[i]
        if dp and eps > 0:
            comm_edges[i] = [
                max(
                    0.0,
                    len(friends.intersection(set(x))) + rng.laplace(0.0, 2.0 / (eps)),
                )
                for x in comms
            ]

        if np.sum(comm_edges[i]) == 0.0:
            ind = rng.choice(len(comms), 1)[0]
            comm_edges[i][ind] = 1.0

    return comm_edges


def get_comm_edges(from_ind, to_ind, adjacency, comms, dp, eps, rng):
    comm_edges = {}
    comm_edges_nondp = {}
    adj_ind = 0
    for i in range(from_ind, to_ind):
        friends = set(np.where(adjacency[adj_ind] > 0)[0])
        comm_edges_nondp[i] = [len(friends.intersection(set(x))) for x in comms]
        comm_edges[i] = comm_edges_nondp[i]
        if dp and eps > 0:
            comm_edges[i] = [
                max(
                    0.0,
                    len(friends.intersection(set(x))) + rng.laplace(0.0, 2.0 / (eps)),
                )
                for x in comms
            ]

        if np.sum(comm_edges[i]) == 0.0:
            ind = rng.choice(len(comms), 1)[0]
            comm_edges[i][ind] = 1.0
        adj_ind += 1

    return comm_edges


def getCommunityCountsMP(
    adjacency, labels, num_classes, rng, dp=False, eps=2.0, num_proc=5
):
    comm_dict = {}
    # This part is fast, no need to parallelize
    for i in range(len(adjacency)):
        if not labels[i] in comm_dict:
            comm_dict[labels[i]] = []
        comm_dict[labels[i]].append(i)
    for i in range(num_classes):
        if not i in comm_dict:
            comm_dict[i] = []
    comms = [val for key, val in sorted(comm_dict.items(), key=operator.itemgetter(0))]

    tasks = []
    from_ind = 0
    to_ind = 0
    slice_size = len(adjacency) // num_proc
    left_over = len(adjacency) % slice_size

    batches = []
    for i in range(num_proc):
        to_ind = from_ind + slice_size
        if i == num_proc - 1:
            if left_over < slice_size / 2:
                batches.append((from_ind, to_ind + left_over))
                continue
        batches.append((from_ind, to_ind))
        from_ind = to_ind

    if left_over >= slice_size // 2:
        batches.append((from_ind, from_ind + left_over))

    # start_time = time.time()
    pool = multiprocessing.Pool(num_proc)
    for batch in batches:
        from_ind, to_ind = batch[0], batch[1]
        tasks.append(
            pool.apply_async(
                get_comm_edges,
                args=(
                    from_ind,
                    to_ind,
                    adjacency[from_ind:to_ind, :],
                    comms,
                    dp,
                    eps,
                    rng,
                ),
            )
        )

    pool.close()
    pool.join()
    # print("Time for pool {} s".format(time.time() - start_time))

    comm_edges = {}
    for task in tasks:
        comm_edges_w_dp = task.get()
        for node in comm_edges_w_dp:
            comm_edges[node] = comm_edges_w_dp[node]

    return comm_edges
