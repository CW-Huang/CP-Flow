# -*- coding: utf-8 -*-
"""
CP-Flow on toy conditional distributions
"""

import gc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from cpflows.flows import SequentialFlow, DeepConvexFlow, ActNorm
from cpflows.icnn import PICNN as PICNN
from data.toy_data import OneDMixtureOfGaussians as ToyData
from cpflows.utils import makedirs


makedirs('figures/toy/cond_MoG/')


def savefig(fn):
    plt.savefig(f'figures/toy/cond_MoG/{fn}')


torch.set_default_dtype(torch.float64)


batch_size_train = 128
batch_size_test = 64

# noinspection PyUnresolvedReferences
train_loader = torch.utils.data.DataLoader(
    ToyData(50000),
    batch_size=batch_size_train, shuffle=True)
# noinspection PyUnresolvedReferences
test_loader = torch.utils.data.DataLoader(
    ToyData(10000),
    batch_size=batch_size_test, shuffle=True)


dimx = 1
dimc = 1
nblocks = 1
depth = 10
k = 64
lr = 0.001
factor = 0.5
patience = 2000
num_epochs = 10
print_every = 100


icnns = [PICNN(dimx, k, dimc, depth, symm_act_first=True, softplus_type='gaussian_softplus',
               zero_softplus=True) for _ in range(nblocks)]

layers = [None] * (2 * nblocks + 1)
layers[0::2] = [ActNorm(dimx) for _ in range(nblocks + 1)]
layers[1::2] = [DeepConvexFlow(icnn, dimx, unbiased=False) for _, icnn in zip(range(nblocks), icnns)]
flow = SequentialFlow(layers)

optim = torch.optim.Adam(flow.parameters(), lr=lr)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(optim, num_epochs * len(train_loader), 0)


cuda = torch.cuda.is_available()
if cuda:
    flow = flow.cuda()

loss_acc = 0
t = 0
grad_norm = 0


for e in range(num_epochs):
    for x in train_loader:
        x, y = x[:, :1], x[:, 1:]
        x = x.double()
        y = y.double()
        if cuda:
            x = x.cuda()

        loss = - flow.logp(x, y).mean()
        optim.zero_grad()
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(flow.parameters(), max_norm=10).item()

        optim.step()
        sch.step()

        loss_acc += loss.item()
        del loss
        gc.collect()
        torch.clear_autocast_cache()

        t += 1
        if t == 1:
            print('init loss:', loss_acc, grad_norm)

        if t % print_every == 0:
            print(t, loss_acc / print_every, grad_norm)
            loss_acc = 0


flow.eval()

colors = sns.color_palette("coolwarm", 9)
fig = plt.figure()
ax = fig.add_subplot()
for f in flow.flows[1::2]:
    f.no_bruteforce = False
xx = torch.linspace(-5, 5, 1000).unsqueeze(1)
for pi, c in zip(np.linspace(0.1, 0.9, 9), colors):
    p = torch.exp(flow.logp(xx, context=torch.ones_like(xx)*pi)).data.numpy()
    plt.plot(xx, p, '--', c=c)

for pi, c in zip(np.linspace(0.1, 0.9, 9), colors):
    p = torch.exp(train_loader.dataset.logp(xx, torch.ones_like(xx)*pi)).data.numpy()
    plt.plot(xx, p, '-', c=c, label='{:.1f}'.format(pi))
plt.legend(loc=2, fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=12)
savefig('1dMOG.png')
