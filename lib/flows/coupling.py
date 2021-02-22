import torch
import torch.nn as nn

__all__ = ['MaskedCouplingBlock']


# noinspection PyUnusedLocal, PyMethodMayBeStatic
class MaskedCouplingBlock(nn.Module):
    """Coupling layer for images implemented using masks.
    """

    def __init__(self, dim, nnet, mask_type='checkerboard0'):
        nn.Module.__init__(self)
        self.d = dim
        self.nnet = nnet
        self.mask_type = mask_type

    def func_s_t(self, x):
        f = self.nnet(x)
        s = torch.sigmoid(f[:, :self.d] + 2.)
        t = f[:, self.d:]
        return s, t

    def forward_transform(self, x, logpx=None):
        # get mask
        b = get_mask(x, mask_type=self.mask_type)

        # masked forward
        x_a = b * x
        s, t = self.func_s_t(x_a)
        y = (x * s + t) * (1 - b) + x_a

        if logpx is None:
            return y
        else:
            return y, logpx + self._logdetgrad(s, b)

    def reverse(self, y, logpy=None, **kwargs):
        # get mask
        b = get_mask(y, mask_type=self.mask_type)

        # masked forward
        y_a = b * y
        s, t = self.func_s_t(y_a)
        x = y_a + (1 - b) * (y - t) / s

        if logpy is None:
            return x
        else:
            return x, logpy - self._logdetgrad(s, b)

    def _logdetgrad(self, s, mask):
        return torch.log(s).mul_(1 - mask).view(s.shape[0], -1).sum(1)

    def extra_repr(self):
        return 'dim={d}, mask_type={mask_type}'.format(**self.__dict__)


# noinspection PyPep8Naming
def _get_checkerboard_mask(x, swap=False):
    n, c, h, w = x.size()

    H = ((h - 1) // 2 + 1) * 2  # H = h + 1 if h is odd and h if h is even
    W = ((w - 1) // 2 + 1) * 2

    # construct checkerboard mask
    if not swap:
        mask = torch.Tensor([[1, 0], [0, 1]]).repeat(H // 2, W // 2)
    else:
        mask = torch.Tensor([[0, 1], [1, 0]]).repeat(H // 2, W // 2)
    mask = mask[:h, :w]
    mask = mask.contiguous().view(1, 1, h, w).expand(n, c, h, w).type_as(x.data)

    return mask.to(x)


def _get_channel_mask(x, swap=False):
    n, c, h, w = x.size()
    assert (c % 2 == 0)

    # construct channel-wise mask
    mask = torch.zeros(x.size())
    if not swap:
        mask[:, :c // 2] = 1
    else:
        mask[:, c // 2:] = 1
    return mask.to(x)


def get_mask(x, mask_type=None):
    if mask_type is None:
        return torch.zeros(x.size()).to(x)
    elif mask_type == 'channel0':
        return _get_channel_mask(x, swap=False)
    elif mask_type == 'channel1':
        return _get_channel_mask(x, swap=True)
    elif mask_type == 'checkerboard0':
        return _get_checkerboard_mask(x, swap=False)
    elif mask_type == 'checkerboard1':
        return _get_checkerboard_mask(x, swap=True)
    else:
        raise ValueError('Unknown mask type {}'.format(mask_type))
