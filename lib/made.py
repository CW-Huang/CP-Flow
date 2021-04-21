"""
Copy from https://github.com/karpathy/pytorch-made/blob/master/made.py

Implements Masked AutoEncoder for Density Estimation, by Germain et al. 2015
Re-implementation by Andrej Karpathy based on https://arxiv.org/abs/1502.03509
"""

import numpy as np
import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F


# ------------------------------------------------------------------------------

class MaskedLinear(nn.Linear):
    """ same as Linear except has a configurable mask on the weights """

    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, x):
        return F.linear(x, self.mask * self.weight, self.bias)


# noinspection PyPep8Naming,PyTypeChecker
class MADE(nn.Module):
    def __init__(self, nin, hidden_sizes, nout, num_masks=1, natural_ordering=False, activation=nn.ReLU()):
        """
        nin: integer; number of inputs
        hidden sizes: a list of integers; number of units in hidden layers
        nout: integer; number of outputs, which usually collectively parameterize some kind of 1D distribution
              note: if nout is e.g. 2x larger than nin (perhaps the mean and std), then the first nin
              will be all the means and the second nin will be stds. i.e. output dimensions depend on the
              same input dimensions in "chunks" and should be carefully decoded downstream appropriately.
              the output of running the tests for this file makes this a bit more clear with examples.
        num_masks: can be used to train ensemble over orderings/connections
        natural_ordering: force natural ordering of dimensions, don't use random permutations
        """

        super().__init__()
        self.nin = nin
        self.nout = nout
        self.hidden_sizes = hidden_sizes
        assert self.nout % self.nin == 0, "nout must be integer multiple of nin"

        # define a simple MLP neural net
        self.net = []
        hs = [nin] + hidden_sizes + [nout]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.extend([
                MaskedLinear(h0, h1),
                activation,
            ])
        self.net.pop()  # pop the last ReLU for the output layer
        self.net = nn.Sequential(*self.net)

        # seeds for orders/connectivities of the model ensemble
        self.natural_ordering = natural_ordering
        self.num_masks = num_masks
        self.seed = 0  # for cycling through num_masks orderings

        self.m = {}
        self.update_masks()  # builds the initial self.m connectivity
        # note, we could also precompute the masks and cache them, but this
        # could get memory expensive for large number of masks.

    def update_masks(self):
        if self.m and self.num_masks == 1:
            return  # only a single seed, skip for efficiency
        L = len(self.hidden_sizes)

        # fetch the next seed and construct a random stream
        rng = np.random.RandomState(self.seed)
        self.seed = (self.seed + 1) % self.num_masks

        # sample the order of the inputs and the connectivity of all neurons
        self.m[-1] = np.arange(self.nin) if self.natural_ordering else rng.permutation(self.nin)
        for l in range(L):
            self.m[l] = rng.randint(self.m[l - 1].min(), self.nin - 1, size=self.hidden_sizes[l])

        # construct the mask matrices
        masks = [self.m[l - 1][:, None] <= self.m[l][None, :] for l in range(L)]
        masks.append(self.m[L - 1][:, None] < self.m[-1][None, :])

        # handle the case where nout = nin * k, for integer k > 1
        if self.nout > self.nin:
            k = int(self.nout / self.nin)
            # replicate the mask across the other outputs
            masks[-1] = np.concatenate([masks[-1]] * k, axis=1)

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.net.modules() if isinstance(l, MaskedLinear)]
        for l, m in zip(layers, masks):
            l.set_mask(m)

    def forward(self, x):
        return self.net(x)


class ContextWrapper(nn.Module):
    def __init__(self, layer, dimc, dimy, scaling=True):
        super().__init__()
        self.layer = layer
        self.scaling = scaling
        self.b = nn.Linear(dimc, dimy)
        if scaling:
            self.a_ = nn.Linear(dimc, dimy)
        self.reset_param()

    def reset_param(self):
        self.b.weight.data.uniform_(-0.001, 0.001)
        self.b.bias.data.zero_()
        self.a_.weight.data.uniform_(-0.001, 0.001)
        self.a_.bias.data.zero_().add_(np.log(np.exp(1) - 1))

    # noinspection PyPropertyDefinition
    def a(self, c):
        return F.softplus(self.a_(c))

    def forward(self, x, c):
        y = self.layer(x)
        if self.scaling:
            y *= self.a(c)
        y += self.b(c)
        return y


class CMADE(nn.Module):
    def __init__(self, nin, hidden_sizes, nout, dimc, num_masks=1, natural_ordering=False, activation=nn.ReLU()):
        super().__init__()
        made = MADE(nin, hidden_sizes, nout, num_masks, natural_ordering, activation)
        self.layers = nn.ModuleList()
        for l in made.net:
            if isinstance(l, MaskedLinear):
                self.layers.extend([ContextWrapper(l, dimc, l.mask.size(0), True)])
            else:
                self.layers.extend([l])

    def forward(self, x, c):
        for l in self.layers:
            if isinstance(l, ContextWrapper):
                x = l(x, c)
            else:
                x = l(x)
        return x
