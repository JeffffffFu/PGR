import torch
import torch.nn.functional as F




def compute_matrix_grads(adj, features, labels, model, idx_train, idx_test, device):
    adj = torch.tensor(adj, requires_grad=True).to(device)

    torch.cuda.empty_cache()

    output = model(features, adj)

    loss = F.nll_loss(output, labels)

    torch.cuda.empty_cache()

    matrix_grads = torch.autograd.grad(
        loss, adj, create_graph=False, retain_graph=True, allow_unused=True)

    torch.cuda.empty_cache()

    return matrix_grads[0]

