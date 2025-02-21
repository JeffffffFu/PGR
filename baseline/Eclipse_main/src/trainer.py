import copy
from dataclasses import dataclass
import os
from globals import MyGlobals
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam, lr_scheduler
from tqdm import tqdm
from sklearn.metrics import f1_score, average_precision_score, accuracy_score
from data import LoadData
import utils
import graph_utils
import models


def set_torch_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class RunConfig:  # Later overwritten in the main function. Only declaration and default initialization here.
    learning_rate: float = MyGlobals.lr
    num_epochs: int = 200
    weight_decay: float = 5e-4
    num_warmup_steps: int = 0
    save_each_epoch: bool = False
    save_epoch: int = 50
    output_dir: str = "."
    eps: float = 0.0
    rank: int = 0
    hidden_size: int = 16
    num_hidden: int = 2
    dropout: float = MyGlobals.dropout
    nl: int = 1


class TrainStats(object):
    """
    A class that encapsulates stats about the trained model
    """

    def __init__(
        self,
        run_config,
        dataset,
        model_name,
        all_outs,
        val_losses,
        val_accuracies,
        test_losses,
        test_accuracies,
        best_epochs,
        seeds,
        f1_scores,
        rare_f1_scores=None,
        test_dataset=None,
    ):
        super(TrainStats, self).__init__
        self.all_outs = all_outs
        self.run_config = run_config
        self.validation = {"loss": val_losses, "acc": val_accuracies}
        self.testing = {"loss": test_losses, "acc": test_accuracies}
        self.best_epochs = best_epochs
        self.dataset = dataset
        self.model_name = model_name
        self.f1_scores = f1_scores
        self.rare_f1_scores = rare_f1_scores
        self.seeds = seeds
        self.test_dataset = test_dataset

    def get_best_avg_val(self):
        """
        Returns the best model for the seed
        """
        return np.mean(self.validation["loss"]), np.mean(self.validation["acc"])

    def print_stats(self):
        print("Best epochs: {}".format(self.best_epochs))
        print(
            "Best val_loss, val_acc {} {}".format(
                self.validation["loss"], self.validation["acc"]
            )
        )
        f1_scores = np.stack(self.f1_scores, axis=1)
        print(
            f"\nPerformance on {self.dataset.name}:\n- "
            f"test accuracy = {np.mean(self.testing['acc']):.3f} +-"
            f"{np.std(self.testing['acc']):.3f}\n- "
            f"micro f1 score = {np.mean(f1_scores[0]):.3f}  +-"
            f"{np.std(f1_scores[0]):.3f}\n- "
            f"macro f1 score = {np.mean(f1_scores[1]):.3f} +-"
            f"{np.std(f1_scores[1]):.3f}\n- "
            f"weighted f1 score = {np.mean(f1_scores[2]):.3f} +-"
            f"{np.std(f1_scores[2]):.3f}\n"
        )
        if self.rare_f1_scores:
            rare_f1_scores = np.stack(self.rare_f1_scores, axis=1)
            print(
                f"\nPerformance on {self.test_dataset.name}:\n- "
                f"test accuracy = {np.mean(self.testing['acc']):.3f} +-"
                f"{np.std(self.testing['acc']):.3f}\n- "
                f"F1 score = {np.mean(rare_f1_scores[0]):.3f}  +-"
                f"{np.std(rare_f1_scores[0]):.3f}\n- "
                f"Precision = {np.mean(rare_f1_scores[1]):.3f} +-"
                f"{np.std(rare_f1_scores[1]):.3f}\n- "
                f"Recall = {np.mean(rare_f1_scores[2]):.3f} +-"
                f"{np.std(rare_f1_scores[2]):.3f}\n"
                f"AP = {np.mean(rare_f1_scores[3]):.3f} +-"
                f"{np.std(rare_f1_scores[3]):.3f}\n"
            )


def train_gcn_on_dataset(
    run_config,
    dataset: utils.Dataset,
    device,
    print_graphs=False,
    dp=False,
    svd=False,
    seeds=[1],
    test_dataset=None,
):
    eps = run_config.eps
    rank = run_config.rank
    print(
        "Training gcn dataset={}, test_dataset={}, eps={}, dp={}, rank={}, svd={}".format(
            dataset, test_dataset, eps, dp, rank, svd
        )
    )
    all_outputs = []
    test_losses = []
    test_accuracies = []
    val_losses = []
    val_accuracies = []
    f1_scores = []
    rare_f1_scores = []
    best_epochs = []
    is_rare = "twitch" in dataset.value
    for i, seed in enumerate(seeds):
        set_torch_seed(seed)
        rng = np.random.default_rng(seed)
        print(f"We run for seed {seed}")
        data_loader = LoadData(
            dataset,
            dp=dp,
            eps=eps,
            svd=svd,
            rank=rank,
            rng=rng,
            rng_seed=seed,
            test_dataset=test_dataset,
            split_num_for_geomGCN_dataset=i % 10,
        )
        num_classes = data_loader.num_classes
        train_features = data_loader.train_features
        train_labels = data_loader.train_labels

        val_features = data_loader.val_features
        val_labels = data_loader.val_labels

        test_features = data_loader.test_features
        test_labels = data_loader.test_labels

        adjacency_matrix = data_loader.train_adj_csr
        val_adjacency_matrix = data_loader.val_adj_csr
        test_adjacency_matrix = data_loader.test_adj_csr

        if data_loader.test_dataset:
            test_dataset = data_loader.test_dataset
        # Create a model based on the config and input size

        model = models.create_model(
            run_config, utils.Architecture.GCN, train_features.size(1), num_classes
        )

        trainer = Trainer(model, rng, seed=seed)
        val_loss, val_acc, best_epoch = trainer.train(
            dataset,
            train_features,
            train_labels,
            val_features,
            val_labels,
            device,
            run_config,
            additional_matrix=adjacency_matrix,
            val_matrix=val_adjacency_matrix,
            test_dataset=test_dataset,
            has_no_val=data_loader.has_no_val(),
        )

        test_loss, test_acc, f1_score, rare_f1_score = trainer.evaluate(
            test_features, test_labels, device, test_adjacency_matrix, is_rare=is_rare
        )

        if print_graphs:
            utils.plot_graph(
                data_loader.train_adj_orig_csr.toarray(),
                trainer.out_labels,
                dataset.name + "_gcn_graph.png",
            )

        all_outputs.append((trainer.out_labels, trainer.logits.detach().cpu()))
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        f1_scores.append(f1_score)
        best_epochs.append(best_epoch)
        if is_rare:
            rare_f1_scores.append(rare_f1_score)

    train_stats = TrainStats(
        run_config,
        dataset,
        model,
        all_outputs,
        val_losses,
        val_accuracies,
        test_losses,
        test_accuracies,
        best_epochs,
        seeds,
        f1_scores,
        rare_f1_scores,
        test_dataset,
    )
    train_stats.print_stats()

    return train_stats


def train_mlp_on_dataset(
    run_config,
    dataset: utils.Dataset,
    device,
    print_graphs=False,
    seeds=[1],
    test_dataset=None,
):
    print("Training mlp dataset={}, test_dataset={}".format(dataset, test_dataset))
    all_outputs = []
    test_losses = []
    test_accuracies = []
    val_losses = []
    val_accuracies = []
    f1_scores = []
    rare_f1_scores = []
    best_epochs = []
    is_rare = "twitch" in dataset.value
    for i, seed in enumerate(seeds):
        set_torch_seed(seed)
        rng = np.random.default_rng(seed)
        print("We run for seed {}".format(seed))
        data_loader = LoadData(
            dataset,
            rng=rng,
            rng_seed=seed,
            test_dataset=test_dataset,
            split_num_for_geomGCN_dataset=i % 10,
        )

        num_classes = data_loader.num_classes
        train_features = data_loader.train_features
        train_labels = data_loader.train_labels

        val_features = data_loader.val_features
        val_labels = data_loader.val_labels

        test_features = data_loader.test_features
        test_labels = data_loader.test_labels

        if num_classes > 2:
            is_rare = False
        if data_loader.test_dataset:
            test_dataset = data_loader.test_dataset

        model = models.create_model(
            run_config, utils.Architecture.MLP, train_features.size(1), num_classes
        )

        trainer = Trainer(model, rng, seed=seed)
        val_loss, val_acc, best_epoch = trainer.train(
            dataset,
            train_features,
            train_labels,
            val_features,
            val_labels,
            device,
            run_config,
            test_dataset=test_dataset,
            has_no_val=data_loader.has_no_val(),
        )

        test_loss, test_acc, f1_score, rare_f1_score = trainer.evaluate(
            test_features, test_labels, device, is_rare=is_rare
        )

        if print_graphs:
            utils.plot_graph(
                data_loader.train_adj_orig_csr.toarray(),
                trainer.out_labels,
                dataset.name + "_mlp_graph.png",
            )

        all_outputs.append((trainer.out_labels, trainer.logits.detach().cpu()))
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        f1_scores.append(f1_score)
        best_epochs.append(best_epoch)
        if is_rare:
            rare_f1_scores.append(rare_f1_score)
        # print("mlp", trainer.out_labels, test_labels)

    train_stats = TrainStats(
        run_config,
        dataset,
        model,
        all_outputs,
        val_losses,
        val_accuracies,
        test_losses,
        test_accuracies,
        best_epochs,
        seeds,
        f1_scores,
        rare_f1_scores,
        test_dataset,
    )
    train_stats.print_stats()

    return train_stats


def train_mmlp_on_dataset(
    run_config,
    dataset: utils.Dataset,
    device,
    print_graphs=False,
    dp=False,
    seeds=[1],
    test_dataset=None,
):
    all_outputs = []
    test_losses = []
    test_accuracies = []
    val_losses = []
    val_accuracies = []
    f1_scores = []
    rare_f1_scores = []
    best_epochs = []
    is_rare = "twitch" in dataset.value

    eps = run_config.eps
    eps = eps * 1.0 / run_config.nl

    for i, seed in enumerate(seeds):
        set_torch_seed(seed)
        rng = np.random.default_rng(seed)
        print("We run for seed {}".format(seed))
        data_loader = LoadData(
            dataset,
            dp=False,
            rng=rng,
            rng_seed=seed,
            test_dataset=test_dataset,
            split_num_for_geomGCN_dataset=i % 10,
        )
        num_classes = data_loader.num_classes
        train_features = data_loader.train_features
        train_labels = data_loader.train_labels

        val_features = data_loader.val_features
        val_labels = data_loader.val_labels

        test_features = data_loader.test_features
        test_labels = data_loader.test_labels

        orig_adj_np = data_loader.train_adj_orig_csr.toarray()

        if num_classes > 2:
            is_rare = False
        if data_loader.test_dataset:
            test_dataset = data_loader.test_dataset

        eps_c = eps

        mmlp = models.create_model(
            run_config,
            utils.Architecture.MMLP,
            train_features.size(1),
            num_classes,
            device=device,
            adjacency_matrix=orig_adj_np,
            eps=eps_c,
            rng=rng,
        )
        print(
            "Input features {}, num_classes {}".format(
                train_features.size(1), num_classes
            )
        )
        model = mmlp.model_list[0]
        trainer = Trainer(model, rng, seed=seed)
        _, _, best_epoch = trainer.train(
            dataset,
            train_features,
            train_labels,
            val_features,
            val_labels,
            device,
            run_config,
            test_dataset=test_dataset,
            has_no_val=data_loader.has_no_val(),
        )
        best_epochs.append(best_epoch)
        # the test_labels are None for the transfer setting when test_dataset is not None
        # so we compute the accuracy on the training set

        trainer.evaluate(train_features, train_labels, device, is_rare=is_rare)

        if print_graphs:
            utils.plot_graph(
                orig_adj_np,
                trainer.out_labels,
                dataset.name + f"_multimlp_{run_config.nl}_0_graph.png",
            )
        ft_nl = []
        ft_nl.append(trainer.logits.detach().cpu())
        out_labels_train = trainer.out_labels

        ft_nl_val = []
        if data_loader.val_on_new_graph():
            # need validation features for inductive setting
            val_loss, val_acc, _, _ = trainer.evaluate(
                val_features, val_labels, device, is_rare=is_rare
            )
            ft_nl_val.append(trainer.logits.detach().cpu())
            out_labels_val = trainer.out_labels

        for it in range(run_config.nl):
            # print(it, out_labels_train)
            comm_counts_dict = graph_utils.getCommunityCountsMP(
                orig_adj_np, out_labels_train, num_classes, rng, dp, eps_c
            )
            mmlp.communities[it + 1] = comm_counts_dict
            comm_counts = torch.from_numpy(
                np.array([comm_counts_dict[j] for j in comm_counts_dict])
            ).type(torch.float32)
            ft_nl.append(comm_counts)
            features1 = torch.cat(ft_nl, 1)
            print("nl-{} Input features {}".format(it, features1.size(1)))

            features1_val = None
            if data_loader.val_on_new_graph():
                comm_counts_dict_val = graph_utils.getCommunityCountsMP(
                    data_loader.val_adj_orig_csr.toarray(),
                    out_labels_val,
                    num_classes,
                    rng,
                    dp,
                    eps_c,
                )
                comm_counts_val = torch.from_numpy(
                    np.array([comm_counts_dict_val[j] for j in comm_counts_dict_val])
                ).type(torch.float32)
                comm_counts_val.to(device)
                ft_nl_val.append(comm_counts_val)
                features1_val = torch.cat(ft_nl_val, 1)
                print("nl-{} Input features {}".format(it, features1_val.size(1)))
            else:
                features1_val = features1

            model1 = mmlp.model_list[it + 1]

            trainer1 = Trainer(model1, rng, seed=seed)
            val_loss, val_acc, best_epoch = trainer1.train(
                dataset,
                features1,
                train_labels,
                features1_val,
                val_labels,
                device,
                run_config,
                test_dataset=test_dataset,
                has_no_val=data_loader.has_no_val(),
            )
            # Record the best_epoch for each model that makes up the MMLP
            best_epochs.append(best_epoch)
            print("features1.size {}".format(features1.size(1)))

            if data_loader.val_on_new_graph():
                # getting validation dataset logits
                val_loss, val_acc, _, _ = trainer1.evaluate(
                    features1_val, val_labels, device, is_rare=is_rare
                )
                out_logits_val = trainer1.logits.detach().cpu()
                out_labels_val = trainer1.out_labels
                ft_nl_val.append(out_logits_val)

            # getting the train logits after validation since we need them for communities in the next iteration
            # also computing the test_acc and scores -- for the case of not a different test set these are also
            # the final test_acc
            test_loss, test_acc, f1_score, rare_f1_score = trainer1.evaluate(
                features1, train_labels, device, is_rare=is_rare
            )
            out_logits_train = trainer1.logits.detach().cpu()
            out_labels_train = trainer1.out_labels
            ft_nl.append(out_logits_train)

            if print_graphs:
                utils.plot_graph(
                    orig_adj_np,
                    trainer1.out_labels,
                    dataset.name + f"_multimlp_{run_config.nl}_{it+1}_graph.png",
                )
        # print(it, out_labels_train)
        # Saving the community dicts after the training for transductive
        comms_file = None
        if not data_loader.is_inductive():
            comms_file = utils.get_comms_pkl_file_name(
                f"mmlp_nl{run_config.nl}", dataset, seed, test_dataset, run_config
            )
            utils.save_comms_pkl(comms_file, mmlp.communities)
        # set all the models in eval() mode
        mmlp.prepare_for_fwd(
            test_features, data_loader.test_adj_orig_csr.toarray(), comms_file
        )
        test_features = test_features.to(device)
        test_labels = test_labels.to(device)

        outputs, test_loss = mmlp(test_features, test_labels)
        
        preds = F.softmax(outputs, dim=1)
        ignore_label = nn.CrossEntropyLoss().ignore_index
        predicted_label = torch.max(preds, dim=1).indices[test_labels != ignore_label]
        true_labels = test_labels[test_labels != -100]
        # preds = F.softmax(outputs, dim=1)
        # predicted_label = preds.max(1)[1].type_as(test_labels)

        test_acc = torch.mean(
            (true_labels == predicted_label).type(torch.FloatTensor)
        ).item()
        f1_score = lk_f1_score(predicted_label, true_labels)
        if is_rare:
            rare_f1_score = rare_class_f1(outputs, true_labels)

        all_outputs.append((test_labels.detach().cpu(), outputs.detach().cpu()))
        test_accuracies.append(test_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        test_losses.append(test_loss.detach().cpu())

        f1_scores.append(f1_score)
        if is_rare:
            rare_f1_scores.append(rare_f1_score)

    train_stats = TrainStats(
        run_config,
        dataset,
        model.model_name,
        all_outputs,
        val_losses,
        val_accuracies,
        test_losses,
        test_accuracies,
        best_epochs,
        seeds,
        f1_scores,
        rare_f1_scores,
        test_dataset,
    )
    train_stats.print_stats()

    return train_stats


def evaluate_mmlp(mmlp, test_features, test_labels, test_orig_adj_np, device, is_rare):
    # set all the models in eval() mode
    mmlp.prepare_for_fwd(test_features, test_orig_adj_np)

    test_features = test_features.to(device)
    test_labels = test_labels.to(device)

    outputs, test_loss = mmlp(test_features, test_labels)
    preds = F.softmax(outputs, dim=1)
    ignore_label = nn.CrossEntropyLoss().ignore_index
    predicted_label = torch.max(preds, dim=1).indices[test_labels != ignore_label]
    true_labels = test_labels[test_labels != -100]
    # preds = F.softmax(outputs, dim=1)
    # predicted_label = preds.max(1)[1].type_as(test_labels)

    test_acc = torch.mean(
        (true_labels == predicted_label).type(torch.FloatTensor)
    ).item()
    f1_score = lk_f1_score(predicted_label, true_labels)
    rare_f1_score = None
    if is_rare:
        rare_f1_score = rare_class_f1(outputs, true_labels)
    return test_loss, test_acc, f1_score, rare_f1_score


class Trainer:
    def __init__(self, model, rng, seed):
        self.model = model
        self.rng = rng
        self.seed = seed
        self.out_labels = []

    def train(
        self,
        dataset,
        train_features,
        train_labels,
        val_features,
        val_labels,
        device,
        run_config,
        log=False,
        additional_matrix=None,
        val_matrix=None,
        test_dataset=None,
        has_no_val=False,
    ):
        """
        Return the loss and accuracy on the validation set if validation is available.
        Otherwise, the loss and accuracy on training.
        """
        self.model = self.model.to(device)
        train_features = train_features.to(device)
        train_labels = train_labels.to(device)
        if additional_matrix is not None:
            additional_matrix = additional_matrix.to(device)

        optimizer = Adam(
            self.model.parameters(),
            lr=run_config.learning_rate,
            weight_decay=run_config.weight_decay,
        )

        # https://huggingface.co/transformers/_modules/transformers/optimization.html#get_linear_schedule_with_warmup
        def lr_lambda(current_step: int):
            if current_step < run_config.num_warmup_steps:
                return float(current_step) / float(max(1, run_config.num_warmup_steps))
            return max(
                0.0,
                float(run_config.num_epochs - current_step)
                / float(max(1, run_config.num_epochs - run_config.num_warmup_steps)),
            )

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)

        if log:
            print("Training started:")
            print(f"\tNum Epochs = {run_config.num_epochs}")
            print(f"\tSave each epoch = {run_config.save_each_epoch}")

        best_loss, best_model_accuracy = float("inf"), 0
        best_model_state_dict = None
        best_output_dir = None
        best_epoch = None
        train_iterator = tqdm(range(0, int(run_config.num_epochs)), desc="Epoch")

        for epoch in train_iterator:
            self.model.train()
            if additional_matrix is not None:
                outputs = self.model(train_features, additional_matrix, train_labels)
            else:
                outputs = self.model(train_features, train_labels)
            loss = outputs[1]

            self.model.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Inductive without validation
            if has_no_val:
                best_epoch = epoch + 1
                # Adding this to compute the validation accuracy which in this case is the training accuracy
                best_loss, best_model_accuracy, _, _ = self.evaluate(
                    train_features, train_labels, device, additional_matrix
                )
                best_model_state_dict = copy.deepcopy(self.model.state_dict())
                dir_name = utils.get_folder_name(
                    run_config,
                    dataset,
                    self.model.model_name,
                    self.seed,
                    test_dataset=test_dataset,
                )
                best_output_dir = os.path.join(run_config.output_dir, dir_name)
                continue

            val_loss, val_accuracy, _, _ = self.evaluate(
                val_features, val_labels, device, val_matrix
            )

            train_iterator.set_description(
                f"Training loss = {loss.item():.4f}, "
                f"val loss = {val_loss:.4f}, "
                f"val accuracy = {val_accuracy:.2f}"
            )

            save_best_model = val_loss < best_loss
            if save_best_model:
                best_epoch = epoch + 1
                best_loss = val_loss
                best_model_accuracy = val_accuracy
                best_model_state_dict = copy.deepcopy(self.model.state_dict())
            if (epoch + 1) % run_config.save_epoch == 0:
                dir_name = utils.get_folder_name(
                    run_config,
                    dataset,
                    self.model.model_name,
                    self.seed,
                    epoch,
                    test_dataset=test_dataset,
                )
                output_dir = os.path.join(run_config.output_dir, dir_name)
                self.save(output_dir)
            if (
                save_best_model
                or run_config.save_each_epoch
                or epoch + 1 == run_config.num_epochs
            ):
                dir_name = utils.get_folder_name(
                    run_config,
                    dataset,
                    self.model.model_name,
                    self.seed,
                    test_dataset=test_dataset,
                )
                best_output_dir = os.path.join(run_config.output_dir, dir_name)
        if log:
            print(
                f"Best model val CE loss = {best_loss:.4f}, "
                f"best model val accuracy = {best_model_accuracy:.2f}"
            )

        # reloads the best model state dict, bit hacky :P
        self.model.load_state_dict(best_model_state_dict)
        if best_output_dir:
            self.save(best_output_dir)
            print("Best saved at {}".format(best_output_dir))
        return best_loss, best_model_accuracy, best_epoch

    def evaluate(
        self, features, test_labels, device, additional_matrix=None, is_rare=False
    ):
        features = features.to(device)
        test_labels = test_labels.to(device)
        if additional_matrix is not None:
            additional_matrix = additional_matrix.to(device)

        self.model.eval()
        if additional_matrix is not None:
            outputs = self.model(features, additional_matrix, test_labels)
        else:
            outputs = self.model(features, test_labels)

        ce_loss = outputs[1].item()
        self.logits = outputs[0]

        preds = F.softmax(outputs[0], dim=1)
        ignore_label = nn.CrossEntropyLoss().ignore_index
        self.out_labels = torch.max(preds, dim=1)[1].cpu().numpy()
        predicted_label = torch.max(preds, dim=1).indices[test_labels != ignore_label]
        true_label = test_labels[test_labels != -100]

        accuracy = torch.mean(
            (true_label == predicted_label).type(torch.FloatTensor)
        ).item()

        my_f1_score = lk_f1_score(predicted_label, true_label)
        if is_rare:
            rare_f1_scores = rare_class_f1(self.logits, test_labels)
        else:
            rare_f1_scores = None

        return ce_loss, accuracy, my_f1_score, rare_f1_scores

    def save(self, output_dir):
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        model_path = os.path.join(output_dir, os.path.basename(output_dir) + ".pth")
        torch.save(self.model.state_dict(), model_path)


def lk_f1_score(preds, labels):
    return (
        f1_score(labels.cpu(), preds.detach().cpu(), average="micro"),
        f1_score(labels.cpu(), preds.detach().cpu(), average="macro"),
        f1_score(labels.cpu(), preds.detach().cpu(), average="weighted"),
    )


def rare_class_f1(output, labels):
    # identify the rare class
    ind = [torch.where(labels == 0)[0], torch.where(labels == 1)[0]]
    rare_class = int(len(ind[0]) > len(ind[1]))

    preds = F.softmax(output, dim=1).max(1)

    ap_score = average_precision_score(
        labels.cpu() if rare_class == 1 else 1 - labels.cpu(), preds[0].detach().cpu()
    )

    preds = preds[1].type_as(labels)

    TP = torch.sum(preds[ind[rare_class]] == rare_class).item()
    T = len(ind[rare_class])
    P = torch.sum(preds == rare_class).item()

    if P == 0:
        return (0, 0, 0, 0)

    precision = TP / P
    recall = TP / T
    F1 = 2 * (precision * recall) / (precision + recall)
    return F1, precision, recall, ap_score
