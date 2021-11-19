import numpy as np
# noinspection PyPep8Naming
import torch.nn.functional as F
import torch.nn as nn
import torch
from cpflows.distributions import log_standard_normal
from cpflows.flows import cpflows
from cpflows.made import MADE, CMADE
from cpflows.naf import sigmoid_flow


_scaling_min = 0.001


# noinspection PyUnusedLocal
class ActNorm(torch.nn.Module):
    """ ActNorm layer with data-dependant init."""

    def __init__(self, num_features, logscale_factor=1., scale=1., learn_scale=True):
        super(ActNorm, self).__init__()
        self.initialized = False
        self.num_features = num_features

        self.register_parameter('b', nn.Parameter(torch.zeros(1, num_features, 1), requires_grad=True))
        self.learn_scale = learn_scale
        if learn_scale:
            self.logscale_factor = logscale_factor
            self.scale = scale
            self.register_parameter('logs', nn.Parameter(torch.zeros(1, num_features, 1), requires_grad=True))

    def forward_transform(self, x, logdet=0):
        input_shape = x.size()
        x = x.view(input_shape[0], input_shape[1], -1)

        if not self.initialized:
            self.initialized = True

            # noinspection PyShadowingNames
            def unsqueeze(x):
                return x.unsqueeze(0).unsqueeze(-1).detach()

            # Compute the mean and variance
            sum_size = x.size(0) * x.size(-1)
            b = -torch.sum(x, dim=(0, -1)) / sum_size
            self.b.data.copy_(unsqueeze(b).data)

            if self.learn_scale:
                var = unsqueeze(torch.sum((x + unsqueeze(b)) ** 2, dim=(0, -1)) / sum_size)
                logs = torch.log(self.scale / (torch.sqrt(var) + 1e-6)) / self.logscale_factor
                self.logs.data.copy_(logs.data)

        b = self.b
        output = x + b

        if self.learn_scale:
            logs = self.logs * self.logscale_factor
            scale = torch.exp(logs) + _scaling_min
            output = output * scale
            dlogdet = torch.sum(torch.log(scale)) * x.size(-1)  # c x h

            return output.view(input_shape), logdet + dlogdet
        else:
            return output.view(input_shape), logdet

    def reverse(self, y, **kwargs):
        assert self.initialized
        input_shape = y.size()
        y = y.view(input_shape[0], input_shape[1], -1)
        logs = self.logs * self.logscale_factor
        b = self.b
        scale = torch.exp(logs) + _scaling_min
        x = y / scale - b

        return x.view(input_shape)

    def extra_repr(self):
        return f"{self.num_features}"


# noinspection PyUnusedLocal
class LayerActnorm(torch.nn.Module):

    def __init__(self):
        super(LayerActnorm, self).__init__()
        self.flow = SequentialFlow([Unsqueeze(1), ActNorm(1), Squeeze(1)])

    def forward_transform(self, x, logdet=0):
        return self.flow.forward_transform(x, logdet, None)

    def reverse(self, y, **kargs):
        return self.flow.reverse(y)


class ActNormNoLogdet(ActNorm):

    def forward(self, x):
        return super(ActNormNoLogdet, self).forward_transform(x)[0]


# noinspection PyUnusedLocal
class Unsqueeze(torch.nn.Module):

    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward_transform(self, x, logdet=0):
        return x.unsqueeze(self.dim), logdet

    def reverse(self, x, **kargs):
        return x.squeeze(self.dim)


# noinspection PyUnusedLocal
class Squeeze(torch.nn.Module):

    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward_transform(self, x, logdet=0):
        return x.squeeze(self.dim), logdet

    def reverse(self, x, **kargs):
        return x.unsqueeze(self.dim)


# noinspection PyPep8Naming
class SequentialFlow(torch.nn.Module):

    def __init__(self, flows):
        super(SequentialFlow, self).__init__()
        self.flows = torch.nn.ModuleList(flows)

    def forward_transform(self, x, logdet=0, context=None, extra=None):
        for flow in self.flows:
            if isinstance(flow, cpflows.DeepConvexFlow) or isinstance(flow, NAFDSF):
                x, logdet = flow.forward_transform(x, logdet,
                                                   context=context,
                                                   extra=extra)
            else:
                prev_logdet = logdet
                x, logdet = flow.forward_transform(x, logdet)
                if extra is not None and len(extra) > 0:
                    extra[0] = extra[0] + (logdet - prev_logdet).detach()
        return x, logdet

    def reverse(self, x, **kwargs):
        # noinspection PyTypeChecker
        for flow in self.flows[::-1]:
            x = flow.reverse(x, **kwargs)
        return x

    def logp(self, x, context=None, extra=None):
        z, logdet = self.forward_transform(x, context=context, extra=extra)
        logp0 = log_standard_normal(z).sum(-1)
        if extra is not None and len(extra) > 0:
            extra[0] = extra[0] + logp0.detach()
        return logp0 + logdet

    def plot_logp(self, b=5, n=100):
        """plotting 2D density"""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        x1 = torch.linspace(-b, b, n)
        x2 = torch.linspace(-b, b, n)
        X2, X1 = torch.meshgrid(x1, x2)
        data = torch.cat([X1.flatten().unsqueeze(1), X2.flatten().unsqueeze(1)], 1)
        if torch.cuda.is_available():
            data = data.cuda()
        p = torch.exp(self.logp(data).cpu()).data.numpy()
        plt.imshow(p.reshape(n, n)[::-1], interpolation='gaussian')
        plt.axis('off')


class Reverse(nn.Module):

    def __init__(self, flow):
        super().__init__()
        self.flow = flow

    def forward_transform(self, *args, **kwargs):
        return self.flow.reverse(*args, **kwargs)

    def reverse(self, *args, **kwargs):
        return self.flow.forward_transform(*args, **kwargs)


# noinspection PyMethodMayBeStatic,PyUnusedLocal
class Flatten(nn.Module):

    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward_transform(self, x, logdet=None, **kwargs):
        flat_x = x.reshape(x.shape[0], -1)
        if logdet is None:
            return flat_x
        else:
            return flat_x, logdet

    def reverse(self, flat_x, logdet=None, **kwargs):
        x = flat_x.reshape(flat_x.shape[0], *self.shape)
        if logdet is None:
            return x
        else:
            return x, logdet

    def extra_repr(self):
        return f"original shape={self.shape}"


# noinspection PyUnusedLocal
class SqueezeLayer(nn.Module):

    def __init__(self, downscale_factor):
        super(SqueezeLayer, self).__init__()
        self.downscale_factor = downscale_factor

    def forward_transform(self, x, logdet=None, **kwargs):
        squeeze_x = squeeze(x, self.downscale_factor)
        if logdet is None:
            return squeeze_x
        else:
            return squeeze_x, logdet

    def reverse(self, y, logdet=None, **kwargs):
        unsqueeze_y = unsqueeze(y, self.downscale_factor)
        if logdet is None:
            return unsqueeze_y
        else:
            return unsqueeze_y, logdet


def unsqueeze(x, upscale_factor=2):
    return torch.pixel_shuffle(x, upscale_factor)


def squeeze(x, downscale_factor=2):
    """
    [:, C, H*r, W*r] -> [:, C*r^2, H, W]
    """
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels = in_channels * (downscale_factor**2)

    out_height = in_height // downscale_factor
    out_width = in_width // downscale_factor

    input_view = x.reshape(batch_size, in_channels, out_height, downscale_factor, out_width, downscale_factor)

    output = input_view.permute(0, 1, 3, 5, 2, 4)
    return output.reshape(batch_size, out_channels, out_height, out_width)


# noinspection PyUnusedLocal
class InvertibleLinear(nn.Module):

    def __init__(self, dim):
        super(InvertibleLinear, self).__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.eye(dim)[torch.randperm(dim)])

    def forward_transform(self, x, logdet=None, **kwargs):
        y = F.linear(x, self.weight)
        if logdet is None:
            return y
        else:
            return y, logdet + self._logdetgrad

    def reverse(self, y, **kwargs):
        x = F.linear(y, self.weight.inverse())
        return x

    @property
    def _logdetgrad(self):
        return torch.slogdet(self.weight)[1]

    def extra_repr(self):
        return 'dim={}'.format(self.dim)


# noinspection PyUnusedLocal,PyPep8Naming
class Invertible1x1Conv(nn.Module):

    def __init__(self, dim):
        super(Invertible1x1Conv, self).__init__()
        self.dim = dim

        # Grab the weight and bias from a randomly initialized Conv2d.
        m = nn.Conv2d(dim, dim, kernel_size=1)
        W = m.weight.clone().detach().reshape(dim, dim)
        LU, pivots = torch.lu(W)
        P, _, _ = torch.lu_unpack(LU, pivots)

        s = torch.diag(LU)
        # noinspection PyTypeChecker
        LU = torch.where(torch.eye(dim) == 0, LU, torch.zeros_like(LU))

        self.register_buffer("P", P)
        self.register_buffer("s_sign", torch.sign(s))
        self.register_parameter("s_log", nn.Parameter(torch.log(torch.abs(s) + 1e-3)))
        self.register_parameter("LU", nn.Parameter(LU))

    @property
    def weight(self):
        L = torch.tril(self.LU, -1) + torch.eye(self.dim).to(self.LU)
        U = torch.triu(self.LU, 1) + torch.diagflat(torch.exp(self.s_log) * self.s_sign)
        return torch.mm(self.P, torch.mm(L, U))

    def forward_transform(self, x, logdet=None, **kwargs):
        y = F.conv2d(x, self.weight.view(self.dim, self.dim, 1, 1))
        if logdet is None:
            return y
        else:
            return y, logdet + self._logdetgrad.expand_as(logdet) * x.shape[2] * x.shape[3]

    def reverse(self, y, **kwargs):
        return F.conv2d(y, self.weight.inverse().view(self.dim, self.dim, 1, 1))

    @property
    def _logdetgrad(self):
        return torch.sum(self.s_log)

    def extra_repr(self):
        return 'dim={}'.format(self.dim)


# noinspection PyUnusedLocal
class LinearIAF(nn.Module):

    def __init__(self, dim, natural_ordering=True):
        super(LinearIAF, self).__init__()
        self.made = MADE(dim, [], dim*2, num_masks=1, natural_ordering=natural_ordering, activation=torch.nn.Identity)
        self.made.net[-1].weight.data.uniform_(-0.001, 0.001)
        self.made.net[-1].bias[:dim].data.zero_()
        self.made.net[-1].bias[dim:].data.zero_().add_(np.log(np.exp(1) - 1))

    def forward(self, x):
        return self.forward_transform(x)

    def forward_transform(self, x, logdet=None, **kwargs):
        m, ls = torch.chunk(self.made(x), 2, 1)
        s = torch.nn.functional.softplus(ls)
        y = m + s * x
        if logdet is None:
            return y
        else:
            return y, logdet + torch.log(s + 1e-8).sum(1)


# noinspection PyUnusedLocal
class IAF(nn.Module):

    def __init__(self, dim, dimh=16, num_hidden_layers=2, natural_ordering=True, activation=torch.nn.ReLU()):
        super(IAF, self).__init__()
        self.dim = dim
        self.dimh = dimh
        self.num_hidden_layers = num_hidden_layers
        hidden_sizes = [dimh] * num_hidden_layers
        self.made = MADE(dim, hidden_sizes, dim*2, num_masks=1, natural_ordering=natural_ordering,
                         activation=activation)
        self.made.net[-1].weight.data.uniform_(-0.001, 0.001)
        self.made.net[-1].bias[:dim].data.zero_()
        self.made.net[-1].bias[dim:].data.zero_().add_(np.log(np.exp(1) - 1))

    def forward(self, x):
        return self.forward_transform(x)

    def forward_transform(self, x, logdet=None, **kwargs):
        m, ls = torch.chunk(self.made(x), 2, 1)
        s = torch.nn.functional.softplus(ls)
        y = m + s * x
        if logdet is None:
            return y
        else:
            return y, logdet + torch.log(s + 1e-8).sum(1)


# noinspection PyUnusedLocal
class NAFDSF(nn.Module):

    def __init__(self, dim, dimh=16, num_hidden_layers=2, natural_ordering=True, ndim=4, dimc=0,
                 activation=torch.nn.ReLU()):
        super(NAFDSF, self).__init__()
        self.dim = dim
        self.dimh = dimh
        self.dimc = dimc
        self.ndim = ndim
        self.num_hidden_layers = num_hidden_layers
        hidden_sizes = [dimh] * num_hidden_layers
        if dimc == 0:
            self.made = MADE(dim, hidden_sizes, dim*ndim*3, num_masks=1, natural_ordering=natural_ordering,
                             activation=activation)
            self.made.net[-1].weight.data.uniform_(-0.001, 0.001)
            self.made.net[-1].bias.data.zero_()
            self.made.net[-1].bias[:dim].data.zero_().add_(np.log(np.exp(1) - 1))
        else:
            # note: there's some flexibility in the design of how to condition on the context
            self.context_net = nn.Sequential(
                nn.Linear(dimc, dimh),
                activation
            )
            self.made = CMADE(dim, hidden_sizes, dim*ndim*3, dimc=dimh, num_masks=1, natural_ordering=natural_ordering,
                              activation=activation)
            self.made.layers[-1].layer.weight.data.uniform_(-0.001, 0.001)
            self.made.layers[-1].layer.bias.data.zero_()
            self.made.layers[-1].layer.bias[:dim].data.zero_().add_(np.log(np.exp(1) - 1))

    def forward(self, x):
        return self.forward_transform(x)

    def forward_transform(self, x, logdet=None, context=None, **kwargs):
        if self.dimc == 0:
            params = self.made(x).view(-1, self.ndim*3, self.dim).permute(0, 2, 1)
        else:
            params = self.made(x, self.context_net(context)).view(-1, self.ndim * 3, self.dim).permute(0, 2, 1)
        y, dlogdet = sigmoid_flow(x, 0, self.ndim, params)
        if logdet is None:
            return y
        else:
            return y, logdet + dlogdet
