
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cosine, euclidean, correlation, cityblock
from sklearn.cluster import KMeans
import torch.nn.init as init

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
        if len(ytrain) < 45000:

            outputs = self(torch.Tensor(Xtrain).to(self.device))
            ytrain = torch.tensor([ytrain])
            ytrain = ytrain.squeeze().long().to(self.device)

            loss = self.criterion(outputs, ytrain)

            loss.backward()
            self.optimizer.step()
        else:  # for large-graph
            outputs = self(torch.Tensor(Xtrain).to(self.device))
            ytrain = torch.tensor([ytrain])
            ytrain = ytrain.squeeze().long().to(self.device)

            pos_mask = (ytrain == 1)
            pos_outputs = outputs[pos_mask]
            pos_labels = ytrain[pos_mask]

            neg_mask = (ytrain == 0)
            neg_outputs = outputs[neg_mask]
            neg_labels = ytrain[neg_mask]

            num_pos = len(pos_outputs) * 2
            rand_indices = torch.randperm(len(neg_outputs))[:num_pos]
            selected_neg_outputs = neg_outputs[rand_indices]
            selected_neg_labels = neg_labels[rand_indices]

            balanced_outputs = torch.cat([pos_outputs, selected_neg_outputs], dim=0)
            balanced_labels = torch.cat([pos_labels, selected_neg_labels], dim=0)

            loss = self.criterion(balanced_outputs, balanced_labels)

            loss.backward()
            self.optimizer.step()


    def train_one_epoch2(self, Xtrain, ytrain): #For Full graph
        self.train()
        self.optimizer.zero_grad()
        outputs = self(torch.Tensor(Xtrain).to(self.device))
        ytrain = torch.tensor([ytrain])
        ytrain = ytrain.squeeze().long().to(self.device)

        pos_mask = (ytrain == 1)
        pos_outputs = outputs[pos_mask]
        pos_labels = ytrain[pos_mask]

        neg_mask = (ytrain == 0)
        neg_outputs = outputs[neg_mask]
        neg_labels = ytrain[neg_mask]

        if len(neg_outputs)<len(pos_outputs)*2:
            selected_neg_outputs=neg_outputs
            selected_neg_labels=neg_labels
        else:
            num_pos = round(len(pos_outputs)*2)
            rand_indices = torch.randperm(len(neg_outputs))[:num_pos]
            selected_neg_outputs = neg_outputs[rand_indices]
            selected_neg_labels = neg_labels[rand_indices]

        balanced_outputs = torch.cat([pos_outputs, selected_neg_outputs], dim=0)
        balanced_labels = torch.cat([pos_labels, selected_neg_labels], dim=0)

        loss = self.criterion(balanced_outputs, balanced_labels)

        loss.backward()
        self.optimizer.step()
    def loss_acc(self, Xtest, ytest):
        self.eval()
        ytest = torch.tensor([ytest])
        ytest = ytest.squeeze().long().to(self.device)
        outputs = self(torch.Tensor(Xtest).to(self.device))


        pos_mask = (ytest == 1)
        pos_outputs = outputs[pos_mask]
        pos_labels = ytest[pos_mask]

        neg_mask = (ytest == 0)
        neg_outputs = outputs[neg_mask]
        neg_labels = ytest[neg_mask]
        num_pos = len(pos_outputs) * 2
        rand_indices = torch.randperm(len(neg_outputs))[:num_pos]
        selected_neg_outputs = neg_outputs[rand_indices]
        selected_neg_labels = neg_labels[rand_indices]

        balanced_outputs = torch.cat([pos_outputs, selected_neg_outputs], dim=0)
        balanced_labels = torch.cat([pos_labels, selected_neg_labels], dim=0)

        loss = self.criterion(balanced_outputs, balanced_labels)

        acc = (balanced_outputs.argmax(dim=1) == balanced_labels).sum() / len(balanced_outputs)

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

def attack_model_inference(attack_model,mia_test_feature,device):
    mia_test_feature = np.array(mia_test_feature)
    ss = StandardScaler()
    mia_x_test = ss.fit_transform(mia_test_feature)
    P = attack_model(torch.Tensor(mia_x_test).to(device))
    return P


def attack_model_inference_cluster(attack_model,mia_test_feature,mia_test_label,device):
    from sklearn.metrics import average_precision_score

    mia_test_feature = np.array(mia_test_feature)
    mia_test_label = np.array(mia_test_label)
    ss = StandardScaler()
    mia_x_test = ss.fit_transform(mia_test_feature)
    P = attack_model(torch.Tensor(mia_x_test).to(device))
    P = P.detach().cpu().numpy()
    kmeans = KMeans(n_clusters=2, random_state=0)
    kmeans.fit(P)

    # 聚类结果
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_



    center_0_mean = np.mean(centers[0])
    center_1_mean = np.mean(centers[1])

    if center_0_mean > center_1_mean:
        predicted_labels = (labels == 0).astype(int)
    else:
        predicted_labels = (labels == 1).astype(int)


    mia_test_label_flat = mia_test_label.flatten()


    # Calculate correct count
    correct_predicted_1_count = np.sum((predicted_labels == 1) & (mia_test_label_flat == 1))

    total_predicted_1 = np.sum(predicted_labels == 1)
    mia_test_label_flat_1 = np.sum(mia_test_label_flat == 1)

    # print("predicted_labels:",predicted_labels)

    precision = correct_predicted_1_count / total_predicted_1
    recall= correct_predicted_1_count / mia_test_label_flat_1
    F1=2*precision*recall/(precision+recall)
    print(f"correct_count: {correct_predicted_1_count}, precision: {precision}, recall: {recall}")

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
    initialize_model_weights(attack_model)

    attack_model.criterion = torch.nn.CrossEntropyLoss()
    attack_model.optimizer = torch.optim.Adam(attack_model.parameters(), lr=0.001, weight_decay=1e-5)
    attack_model.to(device)
    attack_model = train_model(attack_model,
                                       mia_x_train, mia_y_train,
                                       max_patient=50)

    return attack_model
def initialize_model_weights(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            init.xavier_uniform_(layer.weight)
            init.constant_(layer.bias, 0)

def topk_confidence_to_TPL(P, mia_test_label, K, K_hat, device):
    """
    计算 TPL 指标。
    Args:
        P: tensor(N, 2)，每一行是节点的置信度分布。
        mia_test_label: 嵌套列表，每个元素是 [tensor(值)]。
        K: int，理论上的 Top-K 值。
        K_hat: int，实际计算的 Top-K 值。
        device: torch.device，计算设备。
    Returns:
        TPL: float，计算的 TPL 指标。
    """
    # 转换 mia_test_label 为一维张量
    if isinstance(mia_test_label, list):
        mia_test_label = torch.tensor([t[0].item() for t in mia_test_label], dtype=torch.long, device=device)
    else:

        mia_test_label = mia_test_label.squeeze().long().to(device)

    P = P.to(device)
    confidence_label_1 = P[:, 1]

    topk_indices = torch.topk(confidence_label_1, K_hat).indices

    predicted_labels = torch.zeros_like(confidence_label_1, dtype=torch.long)
    predicted_labels[topk_indices] = 1

    C = (predicted_labels[topk_indices] == mia_test_label[topk_indices]).sum().item()

    TPL = C / (K + K_hat - C)

    return TPL


def topk_confidence_to_TPL_PGR(P, mia_test_label, K, K_hat, device, Knowledge,test_nodes):

    N=len(test_nodes)
    if isinstance(mia_test_label, list):
        mia_test_label = torch.tensor([t[0].item() for t in mia_test_label], dtype=torch.long, device=device)
    else:
        mia_test_label = mia_test_label.squeeze().long().to(device)

    if isinstance(Knowledge, list):
        Knowledge_mask = torch.tensor([t[0].item() for t in Knowledge], dtype=torch.long, device=device)
    else:
        Knowledge_mask = Knowledge.squeeze().long().to(device)

    P=P.to(device)
    confidence_label_1 = P[:, 1]
    mask = Knowledge_mask.eq(0)
    valid_confidence = confidence_label_1[mask]
    valid_y_test = mia_test_label[mask]
    if valid_confidence.size(0) == 0:
        raise ValueError("Filtered dataset is empty. Check Knowledge mask.")

    K_hat = min(K_hat, valid_confidence.size(0))
    topk_indices = torch.topk(valid_confidence, K_hat).indices

    predicted_labels = torch.zeros_like(valid_confidence, dtype=torch.long)
    predicted_labels[topk_indices] = 1

    C = (predicted_labels[topk_indices] == valid_y_test[topk_indices]).sum().item()

    TPL_1 = C / (K + K_hat - C)
    TPL_2=(K_hat-C)/((N**2/2)-K_hat+C)
    print(TPL_1,TPL_2)
    return max(TPL_1,TPL_2)


def confidence_to_LIA(P, mia_test_label):

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


    mia_test_label = np.array(mia_test_label)

    mia_test_label_flat = mia_test_label.flatten()
    # Calculate correct count
    # correct_count = np.sum(predicted_labels == mia_test_label_flat)
    correct_predicted_1_count = np.sum((predicted_labels == 1) & (mia_test_label_flat == 1))

    total_predicted_1 = np.sum(predicted_labels == 1)
    mia_test_label_flat_1 = np.sum(mia_test_label_flat == 1)


    precision = correct_predicted_1_count / total_predicted_1
    recall = correct_predicted_1_count / mia_test_label_flat_1
    F1 = 2 * precision * recall / (precision + recall)
    print(f"correct_count: {correct_predicted_1_count}, precision: {precision}, recall: {recall}")

    return F1


def train_model(model, train_x, train_y, max_patient=20, display=-1):
    pbar = tqdm(range(500), leave=False, desc=f"Attack {display}" if display != -1 else "")

    opt_loss = 1e10
    patient = max_patient
    for i in pbar:
        model.train_one_epoch2(train_x, train_y)

    return model





