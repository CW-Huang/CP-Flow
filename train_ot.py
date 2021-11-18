# -*- coding: utf-8 -*-
"""
Learning the optimal transport map (between Gaussians) via CP-Flow (comparing to IAF)
"""

import gc
from scipy import linalg
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
from cpflows.flows import SequentialFlow, DeepConvexFlow, LinearIAF
from cpflows.icnn import ICNN3
from cpflows import distributions
from data.toy_data import Gaussian as ToyData
from cpflows.utils import makedirs


makedirs('figures/OT')


def savefig(fn):
    plt.savefig(f'figures/OT/{fn}')


batch_size_train = 128
batch_size_test = 64
dimx = 2
if dimx == 2:
    m = np.array([1.5, 1.0])
    C = np.array([[0.9, -0.75], [-0.75, 0.9]])  # fixed for visualization
else:
    m = None
    C = None

# noinspection PyUnresolvedReferences
train_loader = torch.utils.data.DataLoader(
    ToyData(50000, dimx, m=m, C=C),
    batch_size=batch_size_train, shuffle=True)
# noinspection PyUnresolvedReferences
test_loader = torch.utils.data.DataLoader(
    ToyData(10000, dimx, train_loader.dataset.m, train_loader.dataset.C),
    batch_size=batch_size_test, shuffle=True)


depth = 5
k = 64
lr = 0.05
factor = 0.5
patience = 2000
num_epochs = 2
print_every = 100

results = list()
for flow_type in ['linear_iaf', 'cpflow']:

    if flow_type == 'cpflow':
        icnn = ICNN3(dimx, k, depth, symm_act_first=False, softplus_type='gaussian_softplus', zero_softplus=True)
        flow = SequentialFlow(
          [DeepConvexFlow(icnn, dimx, unbiased=False, bias_w1=-0.0)]
        )
    else:
        flow = SequentialFlow(
            [LinearIAF(dimx)]
        )

    optim = torch.optim.Adam(flow.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(optim, num_epochs * len(train_loader), eta_min=0)

    cuda = torch.cuda.is_available()
    if cuda:
        flow = flow.cuda()

    # init
    for x in train_loader:
        if cuda:
            x = x.cuda()
        flow.logp(x).mean()
        break

    def logp(x_):
        if flow_type == 'linear_iaf':
            z_, logdet = flow.forward_transform(x_, context=None)
        else:
            z_, logdet = flow.flows[0].forward_transform_bruteforce(x_, context=None)
        return distributions.log_normal(z_, torch.zeros_like(x_), torch.zeros_like(x_)).sum(-1) + logdet

    # noinspection PyPep8Naming
    def plot_logp(b_=5, n_=100):
        """plotting 2D density"""
        x1_ = torch.linspace(-b_, b_, n_)
        x2_ = torch.linspace(-b_, b_, n_)
        X2_, X1_ = torch.meshgrid(x1_, x2_)
        data_ = torch.cat([X1_.flatten().unsqueeze(1), X2_.flatten().unsqueeze(1)], 1)
        if torch.cuda.is_available():
            data_ = data_.cuda()
        p = torch.exp(logp(data_).cpu()).data.numpy()
        plt.imshow(p.reshape(n_, n_)[::-1], interpolation='gaussian')
        plt.axis('off')

    def estimate_l2():
        l2 = 0
        count = 0
        for x_test in test_loader:
            x_test = x_test.view(-1, dimx)
            if cuda:
                x_test = x_test.cuda()
            z = flow.flows[0](x_test)
            l2 += torch.sum((x_test - z)**2).item()
            count += x_test.size(0)
        return l2 / count

    # noinspection PyPep8Naming
    def ot(m1, C1, m2=None, C2=None):
        dim = len(m1)
        if m2 is None:
            m2 = np.zeros_like(m1)
        if C2 is None:
            C2 = np.identity(dim)

        w2 = ((m1-m2) ** 2).sum() + np.trace(C1 + C2 - 2 * linalg.sqrtm(np.dot(C1, C2)))
        return w2  # ** 0.5

    gt = ot(train_loader.dataset.m, train_loader.dataset.C)

    loss_acc = 0
    t = 0
    grad_norm = 0

    flow.train()
    init_l2 = estimate_l2()
    entropy = train_loader.dataset.entropy
    losses = list()
    estimates = list()
    for e in range(num_epochs):
        for x in train_loader:
            x = x.view(-1, dimx)
            if cuda:
                x = x.cuda()

            loss = - logp(x).mean()
            optim.zero_grad()
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(flow.parameters(), max_norm=10).item()

            optim.step()
            sch.step()

            loss_acc += loss.item() - entropy
            del loss
            gc.collect()
            torch.clear_autocast_cache()

            t += 1
            if t == 1:
                print('init loss:', loss_acc, grad_norm)
                print('\t', gt, init_l2)

                losses.append(loss_acc)
                estimates.append(init_l2)

            if t % print_every == 0:
                print(t, loss_acc / print_every, grad_norm)
                losses.append(loss_acc / print_every)
                loss_acc = 0
                estimate = estimate_l2()
                print(f'\t ground truth: {gt}, estimate: {estimate}')
                estimates.append(estimate)

    results.append([losses, estimates])

    if dimx == 2:
        # sample
        flow.eval()
        flow.flows[0].no_bruteforce = False
        plt.figure(figsize=(5, 5))
        plot_logp(4, 100)
        plt.tight_layout()
        savefig(f'OT_learned_gaussian_{flow_type}.png')

        plt.figure(figsize=(5, 5))
        num = 200
        # noinspection PyUnresolvedReferences
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, num))
        ind = np.argsort(test_loader.dataset.data[:num, 0])
        x = test_loader.dataset.data[:num, 0][ind]
        y = test_loader.dataset.data[:num, 1][ind]
        plt.scatter(x, y, color=colors)
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        plt.grid()
        plt.tight_layout()
        savefig(f'OT_x_{flow_type}.png')

        plt.figure(figsize=(5, 5))
        inp = test_loader.dataset.data[:num][ind]
        out = flow.flows[0](inp).data.numpy()
        x = out[:, 0]
        y = out[:, 1]
        plt.scatter(x, y, color=colors)
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        plt.grid()
        plt.tight_layout()
        savefig(f'OT_z_{flow_type}.png')

max_loss = 0
ls = ['dashdot', 'dashed']
cs = ['blue', 'orange']
labels = ['IAF', 'CP-Flow']
plt.figure(figsize=(5, 3))
for i in range(2):
    losses, estimates = results[i]
    max_loss = max(max_loss, max(losses))
    plt.plot(losses[0:], estimates[0:], color=cs[i], linestyle=ls[i], label=labels[i], lw=3)
plt.plot([0, max_loss], [gt, gt], linestyle=':', color='red', label='Optimal', lw=3)
plt.grid()
plt.legend(loc=3, fontsize=12)
plt.xlabel(r'$D_{KL}(p||q)$', fontsize=12)
plt.ylabel('Transport cost', fontsize=12)
plt.xlim(0-max_loss/20, max_loss+max_loss/20)
plt.tight_layout()
savefig(f'OT_dimx{dimx}.png')
