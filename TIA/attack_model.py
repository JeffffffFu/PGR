
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from sklearn import metrics
from tqdm import tqdm

def tpr_at_fpr(y_true, y_score, fpr_th):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    ind = np.argmin(np.abs(fpr - fpr_th))
    tpr_res = tpr[ind]
    return tpr_res

class MLP2Layer(nn.Module):
    def __init__(self, in_dim, out_dim, layer_list=[64, 32], device=torch.device('cpu')):
        super(MLP2Layer, self).__init__()
        assert len(layer_list) == 2

        self.fc1 = nn.Linear(in_dim, layer_list[0])
        self.fc2 = nn.Linear(layer_list[0], layer_list[1])
        self.fc3 = nn.Linear(layer_list[1], out_dim)

        self.outdim = out_dim
        self.indim = in_dim

        self.device = torch.device('cpu')
        self.criterion = None
        self.optimizer = None
        self.device = device

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

    def train_one_epoch(self, Xtrain, ytrain):
        self.train()
        self.optimizer.zero_grad()
        outputs = self(torch.Tensor(Xtrain).to(self.device))
        ytrain = torch.tensor([ytrain])
        ytrain = ytrain.squeeze().long().to(self.device)
        loss = self.criterion(outputs, ytrain)
        loss.backward()
        self.optimizer.step()

    def loss_acc(self, Xtest, ytest):
        self.eval()
        ytest = torch.tensor([ytest])
        ytest = ytest.squeeze().long().to(self.device)
        outputs = self(torch.Tensor(Xtest).to(self.device))
        loss = self.criterion(outputs, ytest)
        acc = (outputs.argmax(dim=1) == ytest).sum() / len(outputs)

        return loss.cpu().detach().item(), acc.cpu().detach().item()

    def all_metrics(self, X_target, y_target, verbos=True):
        outputs_target = self(torch.Tensor(X_target).to(self.device)).cpu()

        acc_target = metrics.accuracy_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        prec_target = metrics.precision_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        recall_target = metrics.recall_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        auc_target = metrics.roc_auc_score(y_target, outputs_target.detach().numpy()[:, 1])
        f1_target = metrics.f1_score(y_target, outputs_target.detach().numpy().argmax(axis=1))
        tpr_05fpr = tpr_at_fpr(y_target, outputs_target.detach().numpy()[:, 1], 0.05)
        tpr_10fpr = tpr_at_fpr(y_target, outputs_target.detach().numpy()[:, 1], 0.10)
        tpr_01fpr = tpr_at_fpr(y_target, outputs_target.detach().numpy()[:, 1], 0.01)

        if verbos:
            print("Accuracy = {:.2%}\n Precision = {:.2%} \n Recall = {:.2%}\n AUC = {:.4}\n F1={:.4}\n TPR@ 1%FPR={:.4}\n TPR@ 5%FPR={:.4}\n TPR@ 10%FPR={:.4} ".format(acc_target,
                                                                                                                                                                         prec_target,
                                                                                                                                                                         recall_target,
                                                                                                                                                                         auc_target,
                                                                                                                                                                         f1_target,
                                                                                                                                                                         tpr_01fpr,
                                                                                                                                                                         tpr_05fpr,
                                                                                                                                                                         tpr_10fpr))
        return [acc_target, prec_target, recall_target, auc_target, f1_target, tpr_01fpr, tpr_05fpr, tpr_10fpr]

    def pred_proba(self, X):
        outputs_target = self(torch.Tensor(X).to(self.device)).cpu()
        return outputs_target.detach().numpy()

    def pred(self, X):
        outputs_target = self(torch.Tensor(X).to(self.device)).cpu()
        return outputs_target.detach().numpy().argmax(axis=1)

def attack_model_inference(attack_model,mia_test_feature,mia_test_label,K,K_hat,N,device):
    mia_test_feature = np.array(mia_test_feature)
    mia_test_label = np.array(mia_test_label)
    ss = StandardScaler()
    mia_x_test = ss.fit_transform(mia_test_feature)
    P = attack_model(torch.Tensor(mia_x_test).to(device))
    C = topk_condifence(P, mia_test_label, K, device)
    item1=C / (K + K_hat - C)
    return item1


def attack_model_inference_PGR_worst_case(attack_model,mia_test_feature,mia_test_label,K,K_hat,N,device,Knowledge):
    mia_test_feature = np.array(mia_test_feature)
    mia_test_label = np.array(mia_test_label)
    ss = StandardScaler()
    mia_x_test = ss.fit_transform(mia_test_feature)
    P = attack_model(torch.Tensor(mia_x_test).to(device))
    C = topk_confidence_PGR_worst_case(P, mia_test_label, K_hat, device,Knowledge)
    item1=C / (K + K_hat - C)
    item2=(K - C) / ((N ** 2 - N)/2 - K_hat + C)
    return max(item1 ,item2)

def attack_model_inference_cluster(attack_model,mia_test_feature,mia_test_label,device):
    mia_test_feature = np.array(mia_test_feature)
    mia_test_label = np.array(mia_test_label)
    ss = StandardScaler()
    mia_x_test = ss.fit_transform(mia_test_feature)
    P = attack_model(torch.Tensor(mia_x_test).to(device))
    P = P.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(P)

    labels = kmeans.labels_
    centers = kmeans.cluster_centers_


    center_0_mean = np.mean(centers[0])
    center_1_mean = np.mean(centers[1])

    if center_0_mean > center_1_mean:
        predicted_labels = (labels == 0).astype(int)
    else:
        predicted_labels = (labels == 1).astype(int)


    mia_test_label_flat = mia_test_label.flatten()  # Now shape is (8001,)

    correct_predicted_1_count = np.sum((predicted_labels == 1) & (mia_test_label_flat == 1))

    total_predicted_1 = np.sum(predicted_labels == 1)
    mia_test_label_flat_1 = np.sum(mia_test_label_flat == 1)


    precision = correct_predicted_1_count / total_predicted_1
    recall= correct_predicted_1_count / mia_test_label_flat_1
    F1=2*precision*recall/(precision+recall)

    return F1

def attack_step(mia_x_train, mia_y_train,device):
    mia_x_train=np.array(mia_x_train)
    mia_y_train=np.array(mia_y_train)
    ss = StandardScaler()
    mia_x_train = ss.fit_transform(mia_x_train)



    attack_model = MLP2Layer(in_dim=mia_x_train.shape[1],
                                     out_dim=2,
                                     layer_list=[32, 32],
                                     device=device
                                     )
    attack_model.criterion = torch.nn.CrossEntropyLoss()
    attack_model.optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.001, weight_decay=1e-5)
    attack_model.to(device)
    attack_model = train_model(attack_model,
                                       mia_x_train, mia_y_train,
                                       max_patient=50)

    return attack_model

def topk_condifence(P,mia_y_test,K,device):
    mia_y_test = torch.tensor(mia_y_test).squeeze().long().to(device)

    confidence_label_1 = P[:, 1]

    topk_indices = torch.topk(confidence_label_1, K).indices

    predicted_labels = torch.zeros_like(confidence_label_1, dtype=torch.long)

    predicted_labels[topk_indices] = 1

    correct_predictions = (predicted_labels[topk_indices] == mia_y_test[topk_indices]).sum().item()

    return correct_predictions


def topk_confidence_PGR_worst_case(P, mia_y_test, K, device, Knowledge):
    mia_y_test = torch.tensor(mia_y_test).squeeze().long().to(device)
    Knowledge = torch.tensor(Knowledge).squeeze().long().to(device)

    confidence_label_1 = P[:, 1].clone()

    exclusion_indices = torch.nonzero(Knowledge.eq(1), as_tuple=True)  # Use eq() to create a tensor of bool values
    confidence_label_1[exclusion_indices] = float('-inf')

    topk_indices = torch.topk(confidence_label_1, K).indices

    predicted_labels = torch.zeros_like(confidence_label_1, dtype=torch.long)

    predicted_labels[topk_indices] = 1

    correct_predictions = (predicted_labels[topk_indices] == mia_y_test[topk_indices]).sum().item()

    return correct_predictions




def train_model(model, train_x, train_y, max_patient=20, display=-1):
    pbar = tqdm(range(500), leave=False, desc=f"Attack {display}" if display != -1 else "")

    opt_loss = 1e10
    patient = max_patient
    for i in pbar:
        model.train_one_epoch(train_x, train_y)
        train_loss, train_acc = model.loss_acc(train_x, train_y)

    return model




