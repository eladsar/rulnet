#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
import math

from khds import Optimizer
from sklearn.model_selection import train_test_split

from tqdm.notebook import tqdm
import itertools

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from typing import Any, Dict
from rtdl import rtdl
import scipy.special
import sklearn.metrics
import sklearn.preprocessing
import zero

device = torch.device(0)
# device = 'cpu'
# Docs: https://yura52.github.io/zero/0.0.4/reference/api/zero.improve_reproducibility.html
zero.improve_reproducibility(seed=123456)


def preprocess_feature(v, nq=20, feature_type=None):
    '''
    get vector of features and calculate
    quantiles/categories

    returns vc, categories

    vc - categorical representation of v
    categories - names of categories (if quantile it is (a, b])

    currently does not handle nan.
    '''

    if type(v) is not pd.Series:
        v = pd.Series(v)

    # for now we use a simple rule to distinguish between categorical and numerical features
    n = v.nunique()

    if n > nq or (feature_type is not None and feature_type == 'numerical'):

        v = v.astype('float')

        c_type = 'numerical'

        q = (np.arange(nq + 1)) / nq

        vc = pd.qcut(v, q, labels=False, duplicates='drop')
        categories = v.quantile(q).values[:-1]
        vc = vc.fillna(-1).values

    else:

        c_type = 'categorical'

        vc, categories = pd.factorize(v)

    # allocate nan value
    categories = np.insert(categories, 0, np.nan)
    vc = vc + 1

    return vc, categories, c_type


def preprocess_table(df, nq=20, offset=True, feature_type=None):
    metadata = defaultdict(OrderedDict)
    n = 0
    dfc = {}

    for c in df.columns:

        if feature_type is not None:
            ft = feature_type[c]
        vc, categories, c_type = preprocess_feature(df[c], nq=nq, feature_type=ft)

        m = len(categories)
        metadata['n_features'][c] = m
        metadata['categories'][c] = categories
        metadata['aggregated_n_features'][c] = n
        metadata['c_type'][c] = c_type

        if offset:
            vc = vc + n

        n = n + m
        dfc[c] = vc

    dfc = pd.DataFrame(dfc).astype(np.int64)

    metadata['total_features'] = n

    return dfc, metadata


# In[3]:


dataset = 'year'
X_num = {}
y = {}
for part in ['train', 'test', 'val']:
    X_num[part] = np.load(f'data/data/{dataset}/N_{part}.npy')
    y[part] = np.load(f'data/data/{dataset}/y_{part}.npy')

if os.path.exists(f'data/data/{dataset}/C_train.npy'):
    X_cat = {}
    X = {}
    for part in ['train', 'test', 'val']:
        X_cat[part] = np.load(f'data/data/{dataset}/C_{part}.npy')
        X[part] = np.concatenate([X_num[part], X_cat[part]], axis=1)
else:
    X = X_num

# idx_test = np.load(f'data/data/{dataset}/idx_test.npy')

y_all = np.concatenate(list(y.values()))

X_all = np.concatenate(list(X.values()))

df = pd.DataFrame(X_all)

# In[4]:


task_type = 'quantile_regression'
out_quantiles = 10

assert task_type in ['binclass', 'multiclass', 'regression', 'quantile_regression']
# y_all = data['class'].astype('float32' if task_type == 'regression' else 'int64')

if task_type == 'multiclass':
    n_classes = int(max(y_all)) + 1
elif task_type == 'quantile_regression':
    n_classes = out_quantiles + 1
else:
    n_classes = None

# !!! CRUCIAL for neural networks when solving regression problems !!!
if task_type == 'regression':
    y_mean = y['train'].mean().item()
    y_std = y['train'].std().item()
    y = {k: (v - y_mean) / y_std for k, v in y.items()}
else:
    y_std = y_mean = None

d_out = n_classes or 1

feature_type = {}
for c in df.columns[:X_num['train'].shape[1]]:
    if c != 2:
        feature_type[c] = 'numerical'
    else:
        feature_type[c] = 'categorical'
for c in df.columns[X_num['train'].shape[1]:]:
    feature_type[c] = 'categorical'
quantiles = 300
dfc, metadata = preprocess_table(df, nq=quantiles, offset=False, feature_type=feature_type)

categorical_columns = [i for i, c in enumerate(metadata['c_type'].values()) if c == 'categorical']
numerical_columns = [i for i, c in enumerate(metadata['c_type'].values()) if c == 'numerical']

x_cat = torch.LongTensor(dfc[categorical_columns].values)
x_num = torch.FloatTensor(df[numerical_columns].values)
n_categories = torch.LongTensor(
    [v - 1 for i, v in enumerate(metadata['n_features'].values()) if i in categorical_columns])
numerical_indices = torch.LongTensor(numerical_columns)
categorical_indices = torch.LongTensor(categorical_columns)

cardinalities = []
if len(cardinalities) == 0:
    cardinalities = None

# In[5]:


X_cat_processed = {}
sizes = []
for v in X.values():
    sizes.append(v.shape[0])
sizes = np.cumsum(sizes)
sizes = np.concatenate(([0], sizes))
for i, part in enumerate(X.keys()):
    X_cat_processed[part] = x_cat[sizes[i]:sizes[i + 1]]

# In[6]:


# x = torch.FloatTensor(X_all)

preprocess = sklearn.preprocessing.QuantileTransformer().fit(X['train'][:, numerical_columns])
X = {
    k: torch.FloatTensor(np.concatenate((preprocess.fit_transform(v), v_cat), axis=1)).to(device)
    for (k, v), (k2, v_cat) in zip(X_num.items(), X_cat_processed.items())
}

y = {k: torch.FloatTensor(v).to(device) for k, v in y.items()}
# cardinalities = [list(metadata['n_features'].values())[k] for k in categorical_columns]


# In[7]:


d_token = 192
k_p = 0.05
k_i = 0
k_d = 0.005
br_d = 0.005
T = 20
model = rtdl.FTTransformer.make_baseline(
    n_num_features=len(numerical_columns),
    cat_cardinalities=cardinalities,
    d_token=d_token,
    n_blocks=3,
    attention_dropout=0.2,
    ffn_d_hidden=256,
    ffn_dropout=0.1,
    residual_dropout=0,
    last_layer_query_idx=[-1],
    d_out=d_out,
    sparse=True,
    attention_transformer_name='MHRuleLayer',
    n_features=X_all.shape[1],
    n_tables=1,
    numerical_indices=numerical_indices,
    categorical_indices=categorical_indices,
    n_quantiles=4,
    n_categories=n_categories,
    emb_dim=d_token,
    momentum=0.001,
    track_running_stats=True,
    initial_mask=0.98,
    k_p=k_p,
    k_i=k_i,
    k_d=k_d,
    T=T,
    clip=br_d,
    use_stats_for_train=True,
    boost=False,
    flatten=False,
    tokenizer=False,
    quantile_embedding=True,
    kaiming_init=True,
    qnorm_flag=False,
    n_rules=32,
)


# In[8]:


class MultipleOptimizer():
    def __init__(self, **op):
        self.optimizers = op
        for k, o in op.items():
            setattr(self, k, o)

    def set_scheduler(self, scheduler, *argc, **argv):

        class MultipleScheduler():
            def __init__(this, scheduler, *argc, **argv):
                for op in self.optimizers.keys():
                    this.schedulers = scheduler(self.optimizers[op], *argc, **argv)

            def step(this, *argc, **argv):
                for op in self.optimizers.keys():
                    this.schedulers.step(*argc, **argv)

        return MultipleScheduler(scheduler, **argv)

    def reset(self):
        for op in self.optimizers.values():
            op.state = defaultdict(dict)

    def zero_grad(self):
        for op in self.optimizers.values():
            op.zero_grad()

    def step(self):
        for op in self.optimizers.values():
            op.step()

    def state_dict(self):
        return {k: op.state_dict() for k, op in self.optimizers.items()}

    def load_state_dict(self, state_dict, state_only=False):

        for k, op in self.optimizers.items():

            if state_only:
                state_dict[k]['param_groups'] = op.state_dict()['param_groups']

            op.load_state_dict(state_dict[k])


def Optimizer(net, dense_ars=None, sparse_args=None, predefined_params=None, module=None):
    if dense_ars is None:
        dense_ars = {'lr': 1e-3, 'eps': 1e-4}
    if sparse_args is None:
        sparse_args = {'lr': 1e-2, 'eps': 1e-4}

    def check_sparse(m):
        return issubclass(type(m), nn.Embedding) and m.sparse

    sparse_parameters = list(
        set(list(itertools.chain(*[m.parameters() for m in net.modules() if check_sparse(m)]))))

    sparse_names = list(
        set(list(itertools.chain([n for n, m in net.named_modules() if check_sparse(m)]))))
    #     print(sparse_names)

    black_list = sparse_names
    if predefined_params is not None:
        black_list += [n for n, p in module.named_parameters()]
    dense_parameters = net.optimization_param_groups(black_list=black_list)
    if predefined_params is not None:
        dense_parameters = dense_parameters + predefined_params
    opt_dense = torch.optim.AdamW(dense_parameters, **dense_ars)

    if len(sparse_parameters) > 0:
        opt_sparse = torch.optim.SparseAdam(sparse_parameters, **sparse_args)
        optimizer = MultipleOptimizer(dense=opt_dense, sparse=opt_sparse)
    else:
        optimizer = MultipleOptimizer(dense=opt_dense)

    return optimizer


# In[9]:


# model.to(device)
# optimizer = Optimizer(model, dense_ars={'lr': 1e-3, 'eps': 1e-8, 'weight_decay': 1e-5},
#                       module = model.better_embedding.emb_num,
#                       predefined_params=[{'params': model.better_embedding.emb_num.parameters(), 'lr': 1e-1, 'eps': 1e-8, 'weight_decay': 0}])
model.to(device)
optimizer = Optimizer(model, dense_ars={'lr': 1e-3, 'eps': 1e-8, 'weight_decay': 1e-5})


# In[10]:


class QuantileRegression(nn.Module):

    def __init__(self, quantiles=20, preprocess=True, data=None):
        super(QuantileRegression, self).__init__()

        self.preprocess = preprocess
        boundaries = None

        self.mu = None
        self.std = None

        if preprocess:

            if len(data.shape) == 1:
                data = data.unsqueeze(-1)
            boundaries = torch.quantile(data, q=torch.arange(quantiles + 1, device=data.device) / quantiles,
                                        dim=0).transpose(0, 1)

            self.q_vals = torch.quantile(data, q=torch.clamp_min(
                (torch.arange(quantiles + 1, device=data.device) - .5) / quantiles, 0), dim=0).transpose(0, 1)
            cond = torch.any(boundaries == self.q_vals.transpose(0, 1), dim=1).unsqueeze(0)
            self.q_vals[cond] = boundaries[cond]

            self.mu = data.mean(dim=0)
            self.std = data.std(dim=0)

        self.qnorm = rtdl.modules.LazyQuantileNorm(quantiles=quantiles, predefined=boundaries, scale=False)

    def forward(self, yhat, y, reduction='mean'):

        if len(y.shape) == 1:
            y = y.unsqueeze(-1)
        y = self.qnorm(y).squeeze(-1)

        if len(yhat.shape) == 3:
            yhat = yhat.transpose(-1, -2)

        loss = F.cross_entropy(yhat, y, reduction=reduction)

        return loss

    def pseudo_accuracy(self, yhat, y, reduction='mean'):
        #     here yhat is the network logits output and we want to calculate the accuracy with respect to the y quantiles

        yhat = yhat.argmax(dim=-1)

        if len(y.shape) == 1:
            y = y.unsqueeze(-1)

        y = torch.searchsorted(self.qnorm.boundaries, y.transpose(0, 1)).transpose(0, 1).squeeze(-1)

        acc = (yhat == y).float()

        if reduction == 'sum':
            acc = acc.sum()
        if reduction == 'mean':
            acc = acc.mean()

        return acc

    def pseudo_mse(self, yhat, y, reduction='mean'):

        yhat = yhat.argmax(dim=-1)
        if len(yhat.shape) == 1:
            yhat = yhat.unsqueeze(-1)

        shape = self.qnorm.boundaries.shape

        #         yhat = torch.gather(self.qnorm.boundaries.expand(len(yhat), *[-1 for _ in shape]), dim=-1, index=yhat.unsqueeze(-1))

        yhat = torch.gather(self.q_vals.expand(len(yhat), *[-1 for _ in shape]), dim=-1, index=yhat.unsqueeze(-1))

        #         yhat = self.values[torch.searchsorted(self.values, yhat)]

        yhat = yhat.squeeze(1).squeeze(1)
        rmse = F.mse_loss(yhat, y, reduction='none')

        #         std = y.std(dim=0) if self.std is None else self.std
        #         mse = mse / std.unsqueeze(0) ** 2

        if reduction == 'sum':
            rmse = rmse.sum()
        if reduction == 'mean':
            rmse = rmse.mean()

        rmse = torch.sqrt(rmse)

        return rmse


# In[11]:


loss_fn = (
    F.binary_cross_entropy_with_logits
    if task_type == 'binclass'
    else F.cross_entropy
    if task_type == 'multiclass'
    else F.mse_loss
    if task_type == 'regression'
    else QuantileRegression(out_quantiles, preprocess=True, data=torch.cat([y['train'], y['val']]))
)

# In[12]:


patience = 16

scheduler = optimizer.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau, mode='min', factor=1 / math.sqrt(10),
                                    patience=patience, threshold=0, threshold_mode='rel', cooldown=0, min_lr=0,
                                    eps=1e-08, verbose=True)
min_lr = 3e-6


# In[13]:


@torch.no_grad()
def evaluate(part, evaluation_type=None):
    evaluation_type = task_type if evaluation_type is None else evaluation_type

    model.eval()
    prediction = []

    for batch in zero.iter_batches((X[part]), batch_size):
        prediction.append(model(batch, ensemble=False))

    prediction = torch.cat(prediction).squeeze(1)
    target = y[part]

    loss = loss_fn(prediction, target)

    if evaluation_type != 'quantile_regression':
        prediction = prediction.cpu().numpy()
        target = target.cpu().numpy()

    if evaluation_type == 'binclass':
        prediction = np.round(scipy.special.expit(prediction))
        score = sklearn.metrics.accuracy_score(target, prediction)
    elif evaluation_type == 'multiclass':
        prediction = prediction.argmax(1)
        score = sklearn.metrics.accuracy_score(target, prediction)
    elif evaluation_type == 'quantile_regression':
        score = loss_fn.pseudo_mse(prediction, target, reduction='mean')
    elif evaluation_type == 'quantile_regression_accuracy':
        score = loss_fn.pseudo_accuracy(prediction, target, reduction='mean')
    else:
        assert evaluation_type == 'regression'
        score = sklearn.metrics.mean_squared_error(target, prediction) ** 0.5 * y_std

    return score, float(loss)


# Create a dataloader for batches of indices
# Docs: https://yura52.github.io/zero/reference/api/zero.data.IndexLoader.html
batch_size = 256
train_loader = zero.data.IndexLoader(len(X['train']), batch_size, device=device)
val_loader = zero.data.IndexLoader(len(X['val']), batch_size, device=device)
# Create a progress tracker for early stopping
# Docs: https://yura52.github.io/zero/reference/api/zero.ProgressTracker.html
progress = zero.ProgressTracker(patience=300)

print(f'Test score before training: {evaluate("test")[0]:.4f}')

# In[14]:


if task_type == 'quantile_regression':

    q = []

    for i in range(len(y['train']) // batch_size + 1):
        q.append(loss_fn.qnorm(y['train'][i * batch_size: (i + 1) * batch_size].unsqueeze(-1)))

    q = torch.cat(q)
    q = q.squeeze(1).long()
    est = loss_fn.q_vals.squeeze(0)[q]

    print(f"Lower bound for the RMSE: {float(torch.sqrt(((y['train'] - est) ** 2).mean()))}")

# In[15]:


n_epochs = 5000
report_frequency = len(X['train']) // batch_size // 5
report_frequency_test = len(X['test']) // batch_size // 2
for epoch in range(1, n_epochs + 1):
    stats = defaultdict(list)
    prediction = []
    y_train = []
    acc = []

    for iteration, batch_idx in tqdm(enumerate(train_loader)):
        model.train()
        optimizer.zero_grad()
        x_batch = X['train'][batch_idx]
        y_batch = y['train'][batch_idx]
        pred = model(x_batch, ensemble=True)
        loss = loss_fn(pred.squeeze(1), y_batch)

        loss_llr = model.better_embedding.get_llr()

        prediction.append(pred.detach())
        y_train.append(y_batch.detach())

        stats['train_loss'].append(float(loss))
        stats['loss_llr'].append(float(loss_llr))

        loss_t = loss + loss_llr
        loss_t.backward()

        optimizer.step()

    prediction = torch.cat(prediction).squeeze(1)
    target = torch.cat(y_train)
    if task_type == 'multiclass':
        prediction = prediction.argmax(1)
        score = sklearn.metrics.accuracy_score(target.cpu().numpy(), prediction.cpu().numpy())
    elif task_type == 'binclass':
        prediction = np.round(scipy.special.expit(prediction))
        score = sklearn.metrics.accuracy_score(target.cpu().numpy(), prediction.cpu().numpy())
    elif task_type == 'quantile_regression':
        score = loss_fn.pseudo_accuracy(prediction, target, reduction='mean')
    else:
        score = 0

    train_score, train_loss = evaluate('train')
    val_score, val_loss = evaluate('val')
    test_score, test_loss = evaluate('test')

    lr = optimizer.dense.param_groups[0]['lr']
    scheduler.step(val_loss)

    print(
        f'Epoch {epoch:03d} | train score: {train_score:.4f} | |  train score with mask: {score:.4f} | | Validation score: {val_score:.4f} | Test score: {test_score:.4f} | train loss: {train_loss:.4f} | | Validation loss: {val_loss:.4f}'
        f"| train loss-llr: {float(np.mean(stats['loss_llr'])):.4f}", end='')

    print(f'bernoulli: {model.better_embedding.br}')
    model.better_embedding.step(train_loss, val_loss)
    progress.update((-1 if task_type in ['regression', 'quantile_regression'] else 1) * val_score)
    if progress.success:
        print(' <<< BEST VALIDATION EPOCH', end='')
        test_acc = test_score
        best_weight = model.state_dict()
        br = model.better_embedding.br
        best_optimizer = optimizer.state_dict()
    print()
    if optimizer.dense.param_groups[0]['lr'] < lr:
        model.load_state_dict(best_weight)
        model.better_embedding.br = br
        optimizer.load_state_dict(best_optimizer, state_only=True)
        print(f'bernoulli: {model.better_embedding.br}')
    if progress.fail:
        print('EAERLY STOPPING')
        break
    if float(optimizer.dense.param_groups[0]['lr']) < min_lr:
        break

# In[24]:


n_epochs = 5000
report_frequency = len(X['train']) // batch_size // 5
report_frequency_test = len(X['test']) // batch_size // 2
for epoch in range(1, n_epochs + 1):
    stats = defaultdict(list)
    prediction = []
    y_train = []
    acc = []

    for iteration, batch_idx in tqdm(enumerate(train_loader)):
        model.train()
        optimizer.zero_grad()
        x_batch = X['train'][batch_idx]
        y_batch = y['train'][batch_idx]
        pred = model(x_batch, ensemble=True)
        loss = loss_fn(pred.squeeze(1), y_batch)

        loss_llr = model.better_embedding.get_llr()

        prediction.append(pred.detach())
        y_train.append(y_batch.detach())

        stats['train_loss'].append(float(loss))
        stats['loss_llr'].append(float(loss_llr))

        loss_t = loss + loss_llr
        loss_t.backward()

        optimizer.step()

    prediction = torch.cat(prediction).squeeze(1)
    target = torch.cat(y_train)
    if task_type == 'multiclass':
        prediction = prediction.argmax(1)
        score = sklearn.metrics.accuracy_score(target.cpu().numpy(), prediction.cpu().numpy())
    elif task_type == 'binclass':
        prediction = np.round(scipy.special.expit(prediction))
        score = sklearn.metrics.accuracy_score(target.cpu().numpy(), prediction.cpu().numpy())
    elif task_type == 'quantile_regression':
        score = loss_fn.pseudo_accuracy(prediction, target, reduction='mean')
    else:
        score = 0

    train_score, train_loss = evaluate('train')
    val_score, val_loss = evaluate('val')
    test_score, test_loss = evaluate('test')

    lr = optimizer.dense.param_groups[0]['lr']
    scheduler.step(val_loss)

    print(
        f'Epoch {epoch:03d} | train score: {train_score:.4f} | |  train score with mask: {score:.4f} | | Validation score: {val_score:.4f} | Test score: {test_score:.4f} | train loss: {train_loss:.4f} | | Validation loss: {val_loss:.4f}'
        f"| train loss-llr: {float(np.mean(stats['loss_llr'])):.4f}", end='')

    print(f'bernoulli: {model.better_embedding.br}')
    model.better_embedding.step(train_loss, val_loss)
    progress.update((-1 if task_type in ['regression', 'quantile_regression'] else 1) * val_score)
    if progress.success:
        print(' <<< BEST VALIDATION EPOCH', end='')
        test_acc = test_score
        best_weight = model.state_dict()
        br = model.better_embedding.br
        best_optimizer = optimizer.state_dict()
    print()
    if optimizer.dense.param_groups[0]['lr'] < lr:
        model.load_state_dict(best_weight)
        model.better_embedding.br = br
        optimizer.load_state_dict(best_optimizer, state_only=True)
        print(f'bernoulli: {model.better_embedding.br}')
    if progress.fail:
        print('EREALY STOPPING')
        break
    if float(optimizer.dense.param_groups[0]['lr']) < min_lr:
        break

# In[16]:


loss_fn.q_vals

# In[17]:


loss_fn.qnorm.boundaries

# In[18]:


pred = prediction.argmax(dim=1)

# In[20]:


pred

# In[49]:


pd.set_option('display.max_rows', 1000)

# In[52]:


target.shape

# In[51]:


pd.value_counts(target.cpu().numpy(), sort=False)

# In[22]:


yhat = prediction
yy = target

# In[25]:


yhat = yhat.argmax(dim=-1)

if len(yy.shape) == 1:
    yy = yy.unsqueeze(-1)

yy = torch.searchsorted(loss_fn.qnorm.boundaries, yy.transpose(0, 1)).transpose(0, 1).squeeze(-1)

acc = (yhat == yy).float()
reduction = 'mean'
if reduction == 'sum':
    acc = acc.sum()
if reduction == 'mean':
    acc = acc.mean()

# In[ ]:


# In[29]:


yhat.shape

# In[34]:


torch.where(yy == 5)[0]

# In[35]:


yhat_c = yhat.to('cpu')

# In[36]:


yy_c = yy.to('cpu')

# In[48]:


pd.value_counts(yy_c.numpy(), sort=False)

# In[59]:


pd.value_counts(yhat_c[torch.where(yy == 10)[0]].numpy(), sort=False)


# In[ ]:


# In[ ]:


def ensambele(model, mode='val', br=0.9, ensambles=128):
    model.better_embedding.br = br
    # ensambles = 1
    # bernoulli = torch.distributions.bernoulli.Bernoulli(probs=1.)

    y_agg = []
    y_hat_agg = []
    model.eval()
    # model = model.to('cpu')

    with torch.no_grad():
        #     for i in tqdm(range(min(1000, int(len(dataset_indices) / (batch_test // ensambles))))):

        for i in tqdm(range(int(len(X[mode]) / (batch_size // ensambles)))):
            #     for iteration, batch_idx in enumerate(test_loader):

            indices = range(i * batch_size // ensambles, (i + 1) * batch_size // ensambles)
            xi = X[mode][indices].unsqueeze(1).repeat(1, ensambles, 1).view(len(indices) * ensambles, -1)
            yi = y[mode][indices]

            yhat = model(xi, ensemble=True)
            #         yhat = torch.softmax(yhat, dim=1)
            #         yhat = np.round(scipy.special.expit(yhat))
            yhat = yhat.view(len(indices), ensambles, -1).mean(dim=1)

            y_hat_agg.append(yhat)
            y_agg.append(yi)

    y_agg = torch.cat(y_agg)

    y_hat_agg = torch.cat(y_hat_agg, dim=0)

    # acc = torch.argmax(y_hat_agg, dim=-1) == y_agg
    acc = (y_hat_agg > 0.0)[:, 0] == y_agg
    return float(acc.float().mean())


# In[ ]:


val_score = {}
for br in tqdm(np.arange(0.7, 1, 0.05)):
    val_score[br] = ensambele(model, mode='val', br=br)
    print(val_score[br])

# In[ ]:


br_max = list(val_score.keys())[np.argmax(list(val_score.values()))]

# In[ ]:


br_max

# In[ ]:


print(ensambele(model, mode='test', br=br_max))

# In[ ]:


# y_pred, label_pred = torch.max(y_hat_agg, dim=1)
bins = 10
y_agg_sorted, y_agg_argsort = y_hat_agg.sort(dim=1, descending=True)

n_labels = y_agg_argsort.shape[-1]

bin_y = np.zeros((n_labels, bins))
bin_acc = np.zeros((n_labels, bins))
ind_n = np.zeros((n_labels, bins))

for k in range(n_labels):

    y_pred = y_agg_sorted[:, k]
    label_pred = y_agg_argsort[:, k]

    asort = torch.argsort(y_pred)

    y_pred = y_pred[asort]
    label_pred = label_pred[asort]
    y_agg_sort = y_agg[asort]

    label_pred = label_pred == y_agg_sort

    n = len(y_pred)
    bins = 10

    for i in range(bins):
        ind = (y_pred > (i / bins)) & (y_pred <= ((i + 1) / bins))
        yi = y_pred[ind]
        acci = label_pred[ind]

        bin_y[k, i] = yi.cpu().numpy().mean()
        bin_acc[k, i] = acci.float().cpu().numpy().mean()
        ind_n[k, i] = ind.float().sum()

    bin_y = np.nan_to_num(bin_y)
    bin_acc = np.nan_to_num(bin_acc)

# In[ ]:


bin_y = (bin_y * ind_n).sum(axis=0) / ind_n.sum(axis=0)
bin_acc = (bin_acc * ind_n).sum(axis=0) / ind_n.sum(axis=0)
ind_n = ind_n.sum(axis=0)
print(f'ERR: {(ind_n * np.abs(bin_y - bin_acc)).sum() / ind_n.sum()}')
plt.bar(np.arange(bins) - 0.25, bin_y)
plt.bar(np.arange(bins) + 0.25, bin_acc)

# In[ ]:


test_loader = zero.data.IndexLoader(len(X_num['test']), batch_size, device=device)

# In[ ]:


# In[ ]:


# In[ ]:


# y_pred, label_pred = torch.max(y_hat_agg, dim=1)
bins = 10
y_agg_sorted, y_agg_argsort = y_hat_agg.sort(dim=1, descending=True)

n_labels = y_agg_argsort.shape[-1]

bin_y = np.zeros((n_labels, bins))
bin_acc = np.zeros((n_labels, bins))
ind_n = np.zeros((n_labels, bins))

for k in range(n_labels):

    y_pred = y_agg_sorted[:, k]
    label_pred = y_agg_argsort[:, k]

    asort = torch.argsort(y_pred)

    y_pred = y_pred[asort]
    label_pred = label_pred[asort]
    y_agg_sort = y_agg[asort]

    label_pred = label_pred == y_agg_sort

    n = len(y_pred)
    bins = 10

    for i in range(bins):
        ind = (y_pred > (i / bins)) & (y_pred <= ((i + 1) / bins))
        yi = y_pred[ind]
        acci = label_pred[ind]

        bin_y[k, i] = yi.cpu().numpy().mean()
        bin_acc[k, i] = acci.float().cpu().numpy().mean()
        ind_n[k, i] = ind.float().sum()

    bin_y = np.nan_to_num(bin_y)
    bin_acc = np.nan_to_num(bin_acc)

# In[ ]:


bin_y = (bin_y * ind_n).sum(axis=0) / ind_n.sum(axis=0)
bin_acc = (bin_acc * ind_n).sum(axis=0) / ind_n.sum(axis=0)
ind_n = ind_n.sum(axis=0)
print(f'ERR: {(ind_n * np.abs(bin_y - bin_acc)).sum() / ind_n.sum()}')
plt.bar(np.arange(bins) - 0.25, bin_y)
plt.bar(np.arange(bins) + 0.25, bin_acc)

# In[ ]:




