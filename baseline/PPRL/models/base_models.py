"""Base model class."""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline.PPRL import manifolds
from baseline.PPRL.layers.layers import FermiDiracDecoder

import baseline.PPRL.models.encoders as encoders
from baseline.PPRL.models.decoders import model2decoder
from baseline.PPRL.utils.eval_utils import acc_f1
from random import shuffle


class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
        self.manifold = getattr(manifolds, self.manifold_name)()
        if self.manifold.name == 'Hyperboloid':
            args.feat_dim = args.feat_dim + 1
        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, args.model)(self.c, args)

    def encode(self, x, adj):
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        h = self.encoder.encode(x, adj)
        return h

    def compute_metrics(self, embeddings, data, split):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError



class subnet(nn.Module):
    def __init__(self, input=32, out=1):
        super(subnet, self).__init__()
        self.fc = nn.Linear(input, out)

    def forward(self, z):
        x = self.fc(z)
        return x


class ADVNCModel(BaseModel):
    """
    Base model for node classification task.
    """

    def __init__(self, args):
        super(ADVNCModel, self).__init__(args)
        self.decoder = model2decoder[args.model](self.c, args)
        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
        if args.pos_weight:
            self.weights = torch.Tensor([1., 1. / data['labels'][idx_train].mean()])
        else:
            self.weights = torch.Tensor([1.] * args.n_classes)
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)

        self.dc = FermiDiracDecoder(r=args.r, t=args.t)
        self.nb_false_edges = args.nb_false_edges
        self.nb_edges = args.nb_edges
        self.net = subnet(32, 32).to(args.device)
        self.mark = 1

        self.encoder = getattr(encoders, args.model)(self.c, args)

    def decode(self, h, adj, idx):
        output = self.decoder.decode(h, adj)
        return F.log_softmax(output[idx], dim=1)

    def decode2(self, h, adj):
        output = self.decoder.decode(h, adj)
        return F.log_softmax(output, dim=1)

    def logist(self, features, adj):
        emb = self.encode(features, adj)
        output = self.decoder.decode(emb, adj)
        return output

    def compute_metrics(self, embeddings, data, split):
        idx = data[f'idx_{split}']
        output = self.decode(embeddings, data['adj_train_norm'], idx)
        loss = F.nll_loss(output, data['labels'][idx], self.weights)
        acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)
        metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        return metrics,output

    def decode1(self, h, idx):
        if self.manifold_name == 'Euclidean':
            h = self.manifold.normalize(h)
        h1 = h.clone()
        emb_in = h1[idx[:, 0], :]
        emb_out = h1[idx[:, 1], :]
        emb = torch.cat((emb_in, emb_out), dim=1)
        weights = self.net(emb)
        weighed_in = weights[:, :16]
        weighed_out = weights[:, 16:]
        if self.mark == 1:
            sqdist = self.manifold.sqdist(weighed_in, weighed_out, self.c)
        else:
            sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        probs = self.dc.forward(sqdist)
        return probs

    def compute_metrics1(self, embeddings, data, split):
        if split == 'train':
            edges_false = data[f'{split}_edges_false'][np.random.randint(0, self.nb_false_edges, self.nb_edges)]
        else:
            edges_false = data[f'{split}_edges_false']
        pos_scores = self.decode1(embeddings, data[f'{split}_edges'])
        neg_scores = self.decode1(embeddings, edges_false)
        loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        loss = loss + F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        # all_scores = torch.cat((pos_scores, neg_scores), 0)
        # shuffle(all_scores)
        # loss_shuffle = F.binary_cross_entropy(all_scores[:pos_scores.shape[0]], torch.ones_like(pos_scores))
        # loss_shuffle = loss_shuffle + F.binary_cross_entropy(all_scores[-neg_scores.shape[0]:], torch.zeros_like(neg_scores))
        # loss_shuffle = loss
        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()
        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
        preds = np.array(preds)
        preds[preds>=0.5] = 1
        preds[preds<0.5] = 0
        preds = list(preds)
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        metrics = {'loss': loss, 'roc': roc, 'ap': ap}
        return metrics


    def init_metric_dict(self):
        return {'acc': -1, 'f1': -1}

    def has_improved(self, m1, m2):
        return m1["f1"] < m2["f1"]

    def save_emb(self):
        torch.save(self.encoder.cpu().state_dict(), 'advnc.pth')
        self.encoder.cuda()

    def load_emb(self):
        params = torch.load('advnc.pth')
        self.encoder.load_state_dict(params)

    def save_net(self):
        torch.save(self.net.cpu().state_dict(), 'advnc.pth')
        self.net.cuda()

    def load_net(self):
        params = torch.load('advnc.pth')
        self.net.load_state_dict(params)

class ADVLPModel(BaseModel):
    """
    Base model for Link prection task.
    """

    def __init__(self, args):
        super(ADVLPModel, self).__init__(args)
        self.decoder = model2decoder[args.model](self.c, args)
        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'
        if args.pos_weight:
            self.weights = torch.Tensor([1., 1. / data['labels'][idx_train].mean()])
        else:
            self.weights = torch.Tensor([1.] * args.n_classes)
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)

        self.dc = FermiDiracDecoder(r=args.r, t=args.t)
        self.nb_false_edges = args.nb_false_edges
        self.nb_edges = args.nb_edges
        self.net = subnet(32, 32).cuda()
        self.mark = 1

        self.encoder = getattr(encoders, args.model)(self.c, args)

    def decode(self, h, adj, idx):
        output = self.decoder.decode(h, adj)
        return F.log_softmax(output[idx], dim=1)

    def decode2(self, h, adj):
        output = self.decoder.decode(h, adj)
        return F.log_softmax(output, dim=1)

    def compute_metrics(self, embeddings, data, split):
        idx = data[f'idx_{split}']
        output = self.decode(embeddings, data['adj_train_norm'], idx)
        loss = F.nll_loss(output, data['labels'][idx], self.weights)
        # output_shuffle = output.clone()
        # shuffle(output_shuffle)
        # loss_shuffle = F.nll_loss(output_shuffle, data['labels'][idx], self.weights)
        acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)
        metrics = {'loss': loss, 'acc': acc, 'f1': f1}
        return metrics,output

    def decode1(self, h, idx):
        if self.manifold_name == 'Euclidean':
            h = self.manifold.normalize(h)
        h1 = h.clone()
        emb_in = h1[idx[:, 0], :]
        emb_out = h1[idx[:, 1], :]
        emb = torch.cat((emb_in, emb_out), dim=1)
        weights = self.net(emb)
        weighed_in = weights[:, :16]
        weighed_out = weights[:, 16:]
        if self.mark == 1:
            sqdist = self.manifold.sqdist(weighed_in, weighed_out, self.c)
        else:
            sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        probs = self.dc.forward(sqdist)
        return probs

    # 链接预测的decoder
    def compute_metrics1(self, embeddings, data, split):
        if split == 'train':
            edges_false = data[f'{split}_edges_false'][np.random.randint(0, self.nb_false_edges, self.nb_edges)]
        else:
            edges_false = data[f'{split}_edges_false']
        pos_scores = self.decode1(embeddings, data[f'{split}_edges'])
        neg_scores = self.decode1(embeddings, edges_false)
        loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        # all_scores = torch.cat((pos_scores, neg_scores), 0)
        # shuffle(all_scores)
        # loss_shuffle = F.binary_cross_entropy(all_scores[:pos_scores.shape[0]], torch.ones_like(pos_scores))
        # loss_shuffle += F.binary_cross_entropy(all_scores[-neg_scores.shape[0]:], torch.zeros_like(neg_scores))
        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()
        labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        preds = list(pos_scores.data.numpy()) + list(neg_scores.data.numpy())
        preds = np.array(preds)
        preds[preds>=0.5] = 1
        preds[preds<0.5] = 0
        preds = list(preds)
        print(preds[:5]+preds[-5:])
        roc = roc_auc_score(labels, preds)
        ap = average_precision_score(labels, preds)
        metrics = {'loss': loss, 'roc': roc, 'ap': ap}
        return metrics

    def init_metric_dict(self):
        return {'roc': -1, 'ap': -1}

    def has_improved(self, m1, m2):
        return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])