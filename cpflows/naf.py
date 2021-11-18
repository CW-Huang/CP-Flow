# noinspection PyPep8Naming
from torch.nn import functional as F
from cpflows.functional import *


def sigmoid_flow(x, logdet=0, ndim=4, params=None, delta=DELTA, logit_end=True):
    """
    element-wise sigmoidal flow described in `Neural Autoregressive Flows` (https://arxiv.org/pdf/1804.00779.pdf)
    :param x: input
    :param logdet: accumulation of log-determinant of jacobian
    :param ndim: number of dimensions of the transform
    :param params: parameters of the transform (batch_size x dimensionality of features x ndim*3 parameters)
    :param delta: small value to deal with numerical stability
    :param logit_end: whether to logit-transform it back to the real space
    :return: transformed data (and log-determinant of jacobian accumulates)
    """
    assert params is not None, 'parameters not provided'
    assert params.size(2) == ndim*3, 'params shape[2] does not match ndim * 3'

    a = act_a(params[:, :, 0 * ndim: 1 * ndim])
    b = act_b(params[:, :, 1 * ndim: 2 * ndim])
    w = act_w(params[:, :, 2 * ndim: 3 * ndim])

    pre_sigm = a * x[:, :, None] + b
    sigm = torch.sigmoid(pre_sigm)
    x_pre = torch.sum(w * sigm, dim=2)

    logj = F.log_softmax(
      params[:, :, 2 * ndim: 3 * ndim], dim=2) + logsigmoid(pre_sigm) + logsigmoid(-pre_sigm) + log(a)
    logj = log_sum_exp(logj, 2).sum(2)
    if not logit_end:
        return x_pre, logj.sum(1) + logdet

    x_pre_clipped = x_pre * (1 - delta) + delta * 0.5
    x_ = log(x_pre_clipped) - log(1 - x_pre_clipped)
    xnew = x_

    logdet_ = logj + np.log(1 - delta) - (log(x_pre_clipped) + log(-x_pre_clipped + 1))
    logdet = logdet_.sum(1) + logdet

    return xnew, logdet
