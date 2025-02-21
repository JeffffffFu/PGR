class MyGlobals(object):

    DATADIR = "../data"
    SVDDIR = "../data/svd/"
    RESULTDIR = "../results"
    LK_DATA = "../data/linkteller-data/"

    nl = -1
    num_seeds = 1
    sample_seed = 42
    cuda_id = 0
    hidden_size = 16
    num_hidden = 2

    # Training args
    eps = 0.0
    with_dp = False
    svd = False
    sort = True
    rank = 0
    lr = 0.05
    num_epochs = 500
    save_epoch = 100
    dropout = 0.1
    weight_decay = 5e-4

    # Attack args
    influence = 0.001
    n_test = 500
