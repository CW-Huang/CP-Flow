# -*- coding: utf-8 -*-
"""
CP-Flow on toy distributions
"""

import gc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
from cpflows.flows import SequentialFlow, DeepConvexFlow, ActNorm, IAF, InvertibleLinear, NAFDSF
from cpflows.icnn import ICNN3
from cpflows import distributions
from data.toy_data import ToyDataset
from PIL import Image
import argparse
from cpflows.utils import makedirs


# parsing arguments

parser = argparse.ArgumentParser('toy')

parser.add_argument('--dataset', type=str, default='EightGaussian',
                    choices=['EightGaussian', 'SwissRoll', 'Rings', 'MAFMoon'])
parser.add_argument('--img_file', type=str, default='')
parser.add_argument('--flow_type', type=str, default='cpflow',
                    choices=['cpflow', 'iaf', 'naf'])
parser.add_argument('--nblocks', type=int, default=1)
parser.add_argument('--depth', type=int, default=20)
parser.add_argument('--dimh', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size_train', type=int, default=128)
parser.add_argument('--batch_size_test', type=int, default=64)
parser.add_argument('--clip_grad', type=float, default=0)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--print_every', type=int, default=100)

args = parser.parse_args()


# read args

cmap = 'BuPu'
prefix = 'toy'
dataset = args.dataset if len(args.img_file) == 0 else args.img_file
flow_type = args.flow_type
img_file = args.img_file
dimx = 2
nblocks = args.nblocks
depth = args.depth
k = args.dimh
lr = args.lr
batch_size_train = args.batch_size_train
batch_size_test = args.batch_size_test
num_epochs = args.num_epochs
print_every = args.print_every
clip_grad = args.clip_grad
plogv = 2 if nblocks == 1 else 0
symm_act_first = True
zero_softplus = True
softplus_type = 'gaussian_softplus'
save = True


if img_file:
    img = np.array(Image.open(img_file).convert('L'))
    h, w = img.shape
    xx = np.linspace(-4, 4, w)
    yy = np.linspace(-4, 4, h)
    xx, yy = np.meshgrid(xx, yy)
    xx = xx.reshape(-1, 1)
    yy = yy.reshape(-1, 1)

    means = np.concatenate([xx, yy], 1)
    # noinspection PyArgumentList
    img = img.max() - img
    probs = img.reshape(-1) / img.sum()

    std = np.array([8 / w / 2, 8 / h / 2])

    class Img2dData(ToyDataset):
        def sample(self, batch_size=200, **kwargs):
            """data and rng are ignored."""
            inds = np.random.choice(int(probs.shape[0]), int(batch_size), p=probs)
            m = means[inds]
            samples = np.random.randn(*m.shape) * std + m
            return torch.from_numpy(samples).float()


    # noinspection PyRedeclaration
    ToyData = Img2dData
else:
    ToyData = ToyDataset.data[dataset]

torch.set_default_dtype(torch.float64)


# noinspection PyUnresolvedReferences
train_loader = torch.utils.data.DataLoader(
    ToyData(50000),
    batch_size=batch_size_train, shuffle=True)
# noinspection PyUnresolvedReferences
test_loader = torch.utils.data.DataLoader(
    ToyData(10000),
    batch_size=batch_size_test, shuffle=True)

folder_name = f'figures/toy/{prefix}_{dataset}'
makedirs(folder_name)


def savefig(fn):
    if save:
        plt.savefig(f'{folder_name}/{fn}')


fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)
spls = train_loader.dataset.sample(100000).data.numpy()
H, _, _ = np.histogram2d(spls[:, 0], spls[:, 1], 200, range=[[-4, 4], [-4, 4]])
plt.imshow(H.T, cmap=cmap)
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
plt.axis('off')
plt.tight_layout()
savefig(f'{ToyData.__name__}_data.png')


if flow_type in ['naf', 'iaf']:
    clip_grad = 10
else:
    clip_grad = 0

if flow_type == 'cpflow':

    icnns = [ICNN3(dimx, k, depth, symm_act_first=symm_act_first, softplus_type=softplus_type,
                   zero_softplus=zero_softplus) for _ in range(nblocks)]
    if nblocks == 1:
        # for printing the potential only
        layers = [None] * (nblocks + 1)
        # noinspection PyTypeChecker
        layers[0] = ActNorm(dimx)
        layers[1:] = [
            DeepConvexFlow(icnn, dimx, unbiased=False, bias_w1=-0.0) for _, icnn in zip(range(nblocks), icnns)]
    else:
        layers = [None] * (2 * nblocks + 1)
        layers[0::2] = [ActNorm(dimx) for _ in range(nblocks + 1)]
        layers[1::2] = [DeepConvexFlow(icnn, dimx, unbiased=False, bias_w1=-0.0,
                                       trainable_w0=False) for _, icnn in zip(range(nblocks), icnns)]
    flow = SequentialFlow(layers)
elif flow_type == 'iaf':
    flows = list()
    flows.extend([ActNorm(dimx)])
    for _ in range(nblocks):
        flows.extend([IAF(dimx, k, depth), InvertibleLinear(dimx), ActNorm(dimx)])
    flow = SequentialFlow(flows)
elif flow_type == 'naf':
    flows = list()
    flows.extend([ActNorm(dimx)])
    for _ in range(nblocks):
        flows.extend([NAFDSF(dimx, k, depth, ndim=16), InvertibleLinear(dimx), ActNorm(dimx)])
    flow = SequentialFlow(flows)
else:
    raise NotImplementedError

print('# parameters', sum([p.numel() for p in flow.parameters()]))

optim = torch.optim.Adam(flow.parameters(), lr=lr)
sch = torch.optim.lr_scheduler.StepLR(optim, 2000, 0.5)

cuda = torch.cuda.is_available()
if cuda:
    flow = flow.cuda()

# init (for actnorm)
for x in train_loader:
    if cuda:
        x = x.cuda()
    flow.logp(x.double()).mean()
    break


def logp(x_):
    z_, logdet = flow.forward_transform(x_, context=None)
    return distributions.log_normal(z_, torch.zeros_like(x_), torch.zeros_like(x_)+plogv).sum(-1) + logdet


# noinspection PyPep8Naming
def plot_logp(b_=5, n_=100, **kwargs):
    """plotting 2D density"""
    x1_ = torch.linspace(-b_, b_, n_)
    x2_ = torch.linspace(-b_, b_, n_)
    X2_, X1_ = torch.meshgrid(x1_, x2_)
    data_ = torch.cat([X1_.flatten().unsqueeze(1), X2_.flatten().unsqueeze(1)], 1)
    if torch.cuda.is_available():
        data_ = data_.cuda()
    p = torch.exp(logp(data_).cpu()).data.numpy()
    plt.imshow(p.reshape(n_, n_), **kwargs)
    plt.axis('off')


loss_acc = 0
t = 0
grad_norm = 0
flow.train()
for e in range(num_epochs):
    for x in train_loader:
        x = x.view(-1, dimx).double()  # TODO: double precision
        if cuda:
            x = x.cuda()

        loss = - logp(x).mean()
        optim.zero_grad()
        loss.backward()

        if clip_grad == 0:
            parameters = [p for p in flow.parameters() if p.grad is not None]
            grad_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2.0) for p in parameters]), 2.0).item()
        else:
            grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(flow.parameters(), clip_grad).item()

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
            print(f'epoch {e} iter {t} loss {loss_acc / print_every} grad norm {grad_norm}')
            loss_acc = 0

# sample
flow.eval()
for f in flow.flows[1::2]:
    f.no_bruteforce = False
fig = plt.figure(figsize=(5, 5))
plot_logp(4, 100, cmap=cmap)
plt.tight_layout()
savefig(f'{ToyData.__name__}_{nblocks}_{flow_type}_{depth}_{k}.png')


# noinspection PyShadowingNames
def interpolate_gif(x, z, steps=100, retain=50, interval=10):

    z_ = (z - z.mean()) / z.std()
    x_ = (x - x.mean()) / x.std()

    fig = plt.figure(figsize=(5, 5))
    plot = plt.scatter(z_[:, 0], -z_[:, 1], s=3, color=colors)
    plt.axis('off')
    plt.tight_layout()

    def animate(step):
        # interpolating from z to x linearly
        print(step)
        step = min(step, steps)
        mid = x_ * step / steps + z_ * (steps - step) / steps
        plot.set_offsets(np.c_[mid[:, 0], -mid[:, 1]])

    # noinspection PyUnresolvedReferences
    ani = matplotlib.animation.FuncAnimation(fig, animate,
                                             frames=steps+retain, interval=interval, repeat=False)
    return ani


# for plotting the potential of cpflow
if nblocks == 1 and flow_type == 'cpflow':
    b = 3.8
    n = 50
    x1 = torch.linspace(-b, b, n)
    x2 = torch.linspace(-b, b, n)
    X2, X1 = torch.meshgrid(x1, x2)
    data = torch.cat([X1.flatten().unsqueeze(1), X2.flatten().unsqueeze(1)], 1)
    if torch.cuda.is_available():
        data = data.cuda()
    logp_ = logp(data)
    data = flow.flows[0].forward_transform(data)[0]
    fl = flow.flows[1]
    x = data.requires_grad_(True)

    F = fl.get_potential(x)
    f = F.cpu().data.numpy()

    # plotting potential
    plt.figure(figsize=(5, 5))
    plt.contour(f.reshape(n, n), levels=20)
    plt.axis('off')
    plt.tight_layout()
    savefig(f'{ToyData.__name__}_{nblocks}_{depth}_{k}_contour.png')

    plt.figure(figsize=(5, 5))
    plt.contourf(f.reshape(n, n), levels=20)
    plt.axis('off')
    plt.tight_layout()
    savefig(f'{ToyData.__name__}_{nblocks}_{depth}_{k}_contourf.png')

    f = torch.autograd.grad(F.sum(), x, create_graph=False)[0]
    f = f.cpu().data.numpy()

    # plotting potential's gradient field
    plt.figure(figsize=(5, 5))
    plt.quiver(f[:, 0].reshape(n, n)[::2, ::2], f[:, 1].reshape(n, n)[::2, ::2],
               torch.exp(logp_).data.numpy().reshape(n, n)[::2, ::2])
    plt.axis('off')
    plt.tight_layout()
    savefig(f'{ToyData.__name__}_{nblocks}_{depth}_{k}_grad.png')

    # plotting gradient map
    plt.figure(figsize=(5, 5))
    plt.plot(f[:, 0], f[:, 1], 'x')
    plt.axis('off')
    plt.tight_layout()

    plt.vlines(b, -b, b, color='red')
    plt.vlines(-b, -b, b, color='red')
    plt.hlines(b, -b, b, color='red')
    plt.hlines(-b, -b, b, color='red')
    savefig(f'{ToyData.__name__}_{nblocks}_{depth}_{k}_z.png')

    plt.figure(figsize=(5, 5))
    data = data.data.numpy()
    fx = data[:, 0].reshape(n, n)
    fy = data[:, 1].reshape(n, n)
    for i in range(n):
        plt.plot(fx[i, :], fy[i, :])
        plt.plot(fx[:, i], fy[:, i])
    plt.axis('off')
    plt.tight_layout()
    savefig(f'{ToyData.__name__}_{nblocks}_{depth}_{k}_x_meshgrid.png')

    plt.figure(figsize=(5, 5))
    fx = f[:, 0].reshape(n, n)
    fy = f[:, 1].reshape(n, n)
    for i in range(n):
        plt.plot(fx[i, :], fy[i, :])
        plt.plot(fx[:, i], fy[:, i])
    plt.axis('off')
    plt.tight_layout()
    savefig(f'{ToyData.__name__}_{nblocks}_{depth}_{k}_z_meshgrid.png')

    # plot convex congjugate
    x_inv = fl.reverse(x)
    F_inv = fl.get_potential(x_inv)
    cc = (x * x_inv).sum(1, keepdim=True) - F_inv

    f = cc.data.numpy()
    # plotting potential
    plt.figure(figsize=(5, 5))
    plt.contour(f.reshape(n, n), levels=20)
    plt.axis('off')
    plt.tight_layout()

    num_samples = 2000
    if num_samples:
        z = torch.randn(num_samples, 2) * np.exp(0.5 * plogv)
        x = fl.reverse(z)
        x = flow.flows[0].reverse(x)
        x = x.data.numpy()
        z = z.data.numpy()

        # noinspection PyUnresolvedReferences
        colors = cm.rainbow(np.linspace(0, 1, num_samples))
        ind = np.argsort((z**2).sum(1))
        x = x[ind]
        z = z[ind]

        plt.figure(figsize=(5, 5))
        plt.scatter(z[:, 0], -z[:, 1], s=3, color=colors)
        plt.axis('off')
        plt.tight_layout()
        savefig(f'{ToyData.__name__}_{nblocks}_{depth}_{k}_z_sample.png')

        plt.figure(figsize=(5, 5))
        plt.scatter(x[:, 0], -x[:, 1], s=3, color=colors)
        plt.axis('off')
        plt.tight_layout()
        savefig(f'{ToyData.__name__}_{nblocks}_{depth}_{k}_x_sample.png')

        ani = interpolate_gif(x, z)

        # noinspection PyUnresolvedReferences
        Writer = matplotlib.animation.writers['ffmpeg']
        writer = Writer(fps=30, bitrate=1800)

        if save:
            ani.save(f'{folder_name}/{ToyData.__name__}_{nblocks}_{depth}_{k}_x_sample.gif', writer=writer)

    # plotting x and z = f(x)
    if ToyData.__name__ == 'EightGaussian':
        # noinspection PyArgumentList
        x, c = ToyData(1).sample(1000, True)
        z = flow.forward_transform(x)[0]
        z = z.data.numpy()
        # noinspection PyUnresolvedReferences
        colors = cm.jet(np.linspace(0, 1, 8))
        plt.figure(figsize=(5, 5))
        for i in range(8):
            plt.scatter(z[c == i, 0], z[c == i, 1], color=colors[i])
        plt.axis('off')
        plt.tight_layout()
        savefig(f'{ToyData.__name__}_{nblocks}_{depth}_{k}_z_encode.png')

        plt.figure(figsize=(5, 5))
        x = x.data.numpy()
        for i in range(8):
            plt.scatter(x[c == i, 0], x[c == i, 1], color=colors[i])
        plt.axis('off')
        plt.tight_layout()
        savefig(f'{ToyData.__name__}_{nblocks}_{depth}_{k}_x_raw.png')
