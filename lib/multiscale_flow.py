import numpy as np
import torch
import torch.nn as nn

from lib.flows import SequentialFlow, ActNorm, SqueezeLayer, Invertible1x1Conv


class MultiscaleFlow(nn.Module):
    """ Creates a stack of flow blocks with squeeze / factor out.
    Main arg:
        block_fn: Function that takes a 3D input shape (c, h, w) and a boolean (fc), returns an invertible block.
    """

    # noinspection PyDefaultArgument
    def __init__(
        self,
        input_size,
        block_fn,
        n_blocks=[16, 16],
        factor_out=True,
        init_layer=None,
        actnorm=False,
        fc_end=False,
        glow=False,
    ):
        super(MultiscaleFlow, self).__init__()
        self.n_scale = len(n_blocks)
        self.n_blocks = n_blocks
        self.factor_out = factor_out
        self.init_layer = init_layer
        self.actnorm = actnorm
        self.fc_end = fc_end
        self.glow = glow

        self.transforms = self._build_net(input_size, block_fn)
        self.dims = [o[1:] for o in self.calc_output_size(input_size)]

    def _build_net(self, input_size, block_fn):
        _, c, h, w = input_size
        transforms = []
        for i in range(self.n_scale):
            transforms.append(
                StackedInvBlocks(
                    block_fn=block_fn,
                    initial_size=(c, h, w),
                    squeeze=(i < self.n_scale - 1),  # don't squeeze last layer
                    init_layer=self.init_layer if i == 0 else None,
                    n_blocks=self.n_blocks[i],
                    actnorm=self.actnorm,
                    fc_end=self.fc_end,
                    glow=self.glow,
                )
            )
            c, h, w = c * 2 if self.factor_out else c * 4, h // 2, w // 2
        return nn.ModuleList(transforms)

    def calc_output_size(self, input_size):
        n, c, h, w = input_size
        if not self.factor_out:
            k = self.n_scale - 1
            return [[n, c * 4**k, h // 2**k, w // 2**k]]
        output_sizes = []
        for i in range(self.n_scale):
            if i < self.n_scale - 1:
                c *= 2
                h //= 2
                w //= 2
                output_sizes.append((n, c, h, w))
            else:
                output_sizes.append((n, c, h, w))
        return tuple(output_sizes)

    def forward(self, x, logdet=0, reverse=False, **kwargs):
        if reverse:
            return self.reverse(x, logdet, **kwargs)
        out = []
        for idx in range(len(self.transforms)):
            x, logdet = self.transforms[idx].forward_transform(x, logdet)

            if self.factor_out and (idx < len(self.transforms) - 1):
                d = x.size(1) // 2
                x, f = x[:, :d], x[:, d:]
                out.append(f)

        out.append(x)
        out = torch.cat([o.view(o.size()[0], -1) for o in out], 1)
        return out, logdet

    def reverse(self, z, logpz=None, **kwargs):
        if self.factor_out:
            z = z.view(z.shape[0], -1)
            zs = []
            i = 0
            for dims in self.dims:
                s = np.prod(dims)
                zs.append(z[:, i:i + s])
                i += s
            zs = [_z.view(_z.size()[0], *zsize) for _z, zsize in zip(zs, self.dims)]

            if logpz is None:
                z_prev = self.transforms[-1].reverse(zs[-1])
                for idx in range(len(self.transforms) - 2, -1, -1):
                    z_prev = torch.cat((z_prev, zs[idx]), dim=1)
                    z_prev = self.transforms[idx].reverse(z_prev, **kwargs)
                return z_prev
            else:
                z_prev, logpz = self.transforms[-1].reverse(zs[-1], logpz, **kwargs)
                for idx in range(len(self.transforms) - 2, -1, -1):
                    z_prev = torch.cat((z_prev, zs[idx]), dim=1)
                    z_prev, logpz = self.transforms[idx].reverse(z_prev, logpz, **kwargs)
                return z_prev, logpz
        else:
            z = z.view(z.shape[0], *self.dims[-1])
            for idx in range(len(self.transforms) - 1, -1, -1):
                if logpz is None:
                    z = self.transforms[idx].reverse(z, **kwargs)
                else:
                    z, logpz = self.transforms[idx].reverse(z, logpz, **kwargs)
            return z if logpz is None else (z, logpz)


class StackedInvBlocks(SequentialFlow):

    def __init__(
        self,
        block_fn,
        initial_size,
        squeeze=True,
        init_layer=None,
        n_blocks=1,
        actnorm=False,
        fc_end=False,
        glow=False,
    ):

        chain = []

        def _actnorm(size, fc):
            if fc:
                return FCWrapper(ActNorm(size[0] * size[1] * size[2]))
            else:
                return ActNorm(size[0])

        def _glow(size, fc):
            if fc:
                raise ValueError("fc invertible layers are disabled due to instability.")
            else:
                return Invertible1x1Conv(size[0])

        if init_layer is not None:
            chain.append(init_layer)
            chain.append(_actnorm(initial_size, fc=False))

        if squeeze:
            # c, h, w = initial_size
            for i in range(n_blocks):
                if glow:
                    chain.append(_glow(initial_size, fc=False))
                if actnorm:
                    chain.append(_actnorm(initial_size, fc=False))
                chain.append(block_fn(i, initial_size, fc=False))
            chain.append(SqueezeLayer(2))
        else:
            for i in range(n_blocks):
                if glow:
                    chain.append(_glow(initial_size, fc=False))
                if actnorm:
                    chain.append(_actnorm(initial_size, fc=False))
                chain.append(block_fn(i, initial_size, fc=False))
            # Use one fully connected block at the end.
            if fc_end:
                # noinspection PyUnboundLocalVariable
                chain.append(FCWrapper(block_fn(i + 1, initial_size, fc=True)))
                if actnorm:
                    chain.append(_actnorm(initial_size, fc=True))

        super(StackedInvBlocks, self).__init__(chain)


class FCWrapper(nn.Module):

    def __init__(self, fc_module):
        super(FCWrapper, self).__init__()
        self.fc_module = fc_module

    def forward_transform(self, x, logdet=None):
        shape = x.shape
        x = x.view(x.shape[0], -1)
        if logdet is None:
            y = self.fc_module.forward_transform(x)
            return y.view(*shape)
        else:
            y, logdet = self.fc_module.forward_transform(x, logdet)
            return y.view(*shape), logdet

    def reverse(self, y, logdet=None, **kwargs):
        shape = y.shape
        y = y.view(y.shape[0], -1)
        if logdet is None:
            x = self.fc_module.reverse(y, **kwargs)
            return x.view(*shape)
        else:
            x, logdet = self.fc_module.reverse(y, logdet, **kwargs)
            return x.view(*shape), logdet
