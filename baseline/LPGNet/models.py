import numpy as np
import torch
import torch.nn as nn
import baseline.LPGNet.utils
import os

from baseline.LPGNet import utils, graph_utils


def create_model(
    run_config,
    arch,
    input_size,
    output_size,
    adjacency_matrix=None,
    device=None,
    eps=0,
    rng=None,
):
    model = None

    if rng is None or adjacency_matrix is None or device is None:
        print("Missing arguments to initialize {}".format(arch))
        exit()
    model = MultiMLP(
        run_config=run_config,
        input_size=input_size,
        output_size=output_size,
        adjacency=adjacency_matrix,
        model_name="mmlp",
        num_hidden=run_config.num_hidden,
        device=device,
        eps=eps,
        rng=rng,
    )

    return model

class MultiMLP(nn.Module):
    def __init__(
        self,
        run_config,
        input_size,
        output_size,
        adjacency,
        device,
        rng,
        model_name="mmlp",
        eps=0.0,
        num_hidden=2,
    ):
        super(MultiMLP, self).__init__()

        print("MMLP input features size {}".format(input_size))
        self.input_size = input_size
        self.model_list = []
        self.model_name = model_name
        print(run_config)
        model = MLP(
            model_name="{}_{}_0".format(model_name, run_config.nl),
            input_size=input_size,
            hidden_size=run_config.hidden_size,
            output_size=output_size,
            dropout=run_config.dropout,
            num_hidden=num_hidden,
        )
        model=model.to(device)
        self.model_list.append(model)
        input_size = 2 * output_size
        for it in range(1, run_config.nl + 1):
            print("MMLP-{} input features size {}".format(it, input_size))
            model = MLP(
                model_name="{}_{}_{}".format(model_name, run_config.nl, it),
                input_size=input_size,
                hidden_size=run_config.hidden_size,
                output_size=output_size,
                dropout=run_config.dropout,
                num_hidden=num_hidden,
            )
            model = model.to(device)
            self.model_list.append(model)
            input_size += 2 * output_size

        self.output_size = output_size
        self.device = device
        self.nl = run_config.nl
        self.eps = eps
        self.adjacency = adjacency
        if eps > 0:
            self.dp = True
        else:
            self.dp = False
        self.communities = {}
        self.rng = rng

    def prepare_for_fwd(self, test_features, test_adjacency, comms_file=None):
        self.adjacency = test_adjacency
        self.input_size = test_features.size(1)
        # The adjacency matrix may not the same as the one used for testing
        # so clear it out
        self.communities = {}
        if (not comms_file == None) and os.path.isfile(comms_file):
            self.communities = utils.load_comms_pkl(comms_file)
        for model in self.model_list:
            model.eval()

    def load_model_from(self, path, device, comms_file=None):
        assert len(path) == len(self.model_list)
        assert self.nl >= 1
        # Goes from 0 to nl
        for i in range(self.nl + 1):
            self.model_list[i].load_state_dict(torch.load(path[i]))
            self.model_list[i].to(device)

        for model in self.model_list:
            model.eval()

        self.communities = {}
        print("Comms file:", comms_file)
        if (not comms_file == None) and os.path.isfile(comms_file):
            self.communities = utils.load_comms_pkl(comms_file)

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None):

        outputs = self.model_list[0](x, labels)
        if labels is not None:
            loss = outputs[1]
            # print("loss {}".format(loss))
            # print("outputs {}".format(outputs[0]))
            outputs = outputs[0]

        ft_nl = []
        ft_nl.append(outputs.detach().cpu())
        for it in range(1, self.nl + 1):
            if it not in self.communities:
                predicted_label_temp = torch.max(outputs, dim=1)
                out_labels = predicted_label_temp[1].cpu().numpy()
                comm_counts_dict = graph_utils.getCommunityCountsMP(
                    self.adjacency,
                    out_labels,
                    self.output_size,
                    self.rng,
                    self.device,
                    self.dp,
                    self.eps,
                )
                self.communities[it] = comm_counts_dict
                # import pickle as pkl
                # with open("comm_counts_dict.pkl", "wb") as f:
                #     pkl.dump(comm_counts_dict, f)
            else:
                comm_counts_dict = self.communities[it]

            comm_counts = torch.from_numpy(
                np.array([comm_counts_dict[j] for j in comm_counts_dict])
            ).type(torch.float32)
            ft_nl.append(comm_counts)
            features = torch.cat(ft_nl, 1)
            features = features.to(self.device)
            outputs = self.model_list[it](features, labels)
            if labels is not None:
                outputs = outputs[0]
                loss = outputs[1]
            ft_nl.append(outputs.detach().cpu())

        if labels is None:
            return outputs
        return outputs, loss



    def logist(self, x: torch.Tensor, adj, labels: torch.Tensor = None):
        # comms_file = None
        #
        # self.prepare_for_fwd(
        #     x, adj, comms_file
        # )
        self.adjacency = adj
        self.input_size = x.size(1)
        self.output_size = x.size(1)
        outputs = self.model_list[0].logist(x)

        ft_nl = []
        ft_nl.append(outputs.detach().cpu())
        for it in range(1, self.nl + 1):
            predicted_label_temp = torch.max(outputs, dim=1)
            out_labels = predicted_label_temp[1].cpu().numpy()

            comm_counts_dict = graph_utils.getCommunityCountsMP(
                self.adjacency,
                out_labels,
                self.output_size,
                self.rng,
                self.device,
                self.dp,
                self.eps,
            )
            self.communities[it] = comm_counts_dict
            # import pickle as pkl
            # with open("comm_counts_dict.pkl", "wb") as f:

            comm_counts = torch.from_numpy(
                np.array([comm_counts_dict[j] for j in comm_counts_dict])
            ).type(torch.float32)
            ft_nl.append(comm_counts)

            features = torch.cat(ft_nl, 1)
            features = features.to(self.device)
            self.model_list[it].input_size=features.size(1)

            model = MLP(
                model_name="{}_{}_0",
                input_size=features.size(1),
                hidden_size=256,
                output_size=labels.max().item() + 1,
                dropout=0.0,
                num_hidden=2,
            )
            model = model.to(self.device)

            outputs = model.logist(features)

            ft_nl.append(outputs.detach().cpu())

        return outputs

class MLP(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        dropout=0.1,
        model_name="mlp",
        num_hidden=2,
    ):
        super(MLP, self).__init__()

        self.model_name = model_name
        self.W1 = nn.Linear(input_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, output_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.num_hidden = num_hidden
        self.model_list = []
        if num_hidden > 2:
            self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
            self.W3 = nn.Linear(hidden_size, output_size, bias=False)

    def load_model_from(self, path, device):
        self.load_state_dict(torch.load(path[0]))
        self.to(device)
        self.eval()

    def forward(self, x: torch.Tensor, labels: torch.Tensor = None):
        x = self.W1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.W2(x)
        if self.num_hidden > 2:
            x = self.relu(x)
            x = self.dropout(x)
            x = self.W3(x)
        if labels is None:
            return x

        loss = nn.CrossEntropyLoss()(x, labels)
        return x, loss

    def logist(self, x: torch.Tensor):
        x = self.W1(x)
        x = self.relu(x)
      #  x = self.dropout(x)
        x = self.W2(x)
        if self.num_hidden > 2:
            x = self.relu(x)
            # x = self.dropout(x)
            x = self.W3(x)

        return x

class TwoLayerGCN(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, dropout=0.1, model_name="2layergcn"
    ):
        super(TwoLayerGCN, self).__init__()

        self.model_name = model_name
        self.W1 = nn.Linear(input_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.model_list = []

    def load_model_from(self, path, device):
        self.load_state_dict(torch.load(path[0]))
        self.to(device)
        self.eval()

    def forward(
        self,
        x: torch.Tensor,
        adjacency_hat: torch.sparse_coo_tensor,
        labels: torch.Tensor = None,
    ):
        x = self.W1(x)
        x = torch.sparse.mm(adjacency_hat, x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.W2(x)
        x = torch.sparse.mm(adjacency_hat, x)
        if labels is None:
            return x

        loss = nn.CrossEntropyLoss()(x, labels)
        return x, loss


class ThreeLayerGCN(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, dropout=0.1, model_name="3layergcn"
    ):
        super(ThreeLayerGCN, self).__init__()

        self.model_name = model_name
        self.W1 = nn.Linear(input_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.W3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.model_list = []

    def load_model_from(self, path, device):
        self.load_state_dict(torch.load(path[0]))
        self.to(device)
        self.eval()

    def forward(
        self,
        x: torch.Tensor,
        adjacency_hat: torch.sparse_coo_tensor,
        labels: torch.Tensor = None,
    ):
        x = self.W1(x)
        x = torch.sparse.mm(adjacency_hat, x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.W2(x)
        x = torch.sparse.mm(adjacency_hat, x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.W3(x)
        x = torch.sparse.mm(adjacency_hat, x)
        if labels is None:
            return x

        loss = nn.CrossEntropyLoss()(x, labels)
        return x, loss
