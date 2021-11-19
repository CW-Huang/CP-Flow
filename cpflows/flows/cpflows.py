import torch
# noinspection PyPep8Naming
import torch.nn.functional as nnF
from functools import partial
import numpy as np
from cpflows.logdet_estimators import stochastic_lanczos_quadrature, \
    unbiased_logdet, stochastic_logdet_gradient_estimator, CG_ITERS_TRACER
import gc
import warnings

HESS_NORM_TRACER = list()


# noinspection PyPep8Naming
class ConvexFlow(torch.nn.Module):
    """
    Shallow convex potential flow with hard-coded logdet of Jacobian (not used in paper).
    """

    def __init__(self, dim=2, k=5, gain=1):
        super(ConvexFlow, self).__init__()
        self.dim = dim
        self.k = k
        self.A = torch.nn.Parameter(torch.randn(dim, k) / np.sqrt(k))
        self.b = torch.nn.Parameter(torch.randn(k) * gain)
        self.w = torch.nn.Parameter(torch.randn(k) * 0.001)
        self.w0 = torch.nn.Parameter(torch.ones(1))
        self.act = torch.sigmoid
        self.dact = lambda x: torch.sigmoid(x) * torch.sigmoid(-x)

    def forward_transform(self, x, logdet=0):
        pre_act = torch.mm(x, self.A) + self.b
        act = self.act(pre_act)
        dact = self.dact(pre_act)

        out = torch.mm(nnF.softplus(self.w) * act, self.A.permute(1, 0)) + nnF.softplus(self.w0) * x
        J = dact[:, None, None] * self.A[:, None] * self.A[None, :] * nnF.softplus(self.w)
        J = J.sum(-1) + nnF.softplus(self.w0) * torch.eye(self.dim).to(x.device)
        return out, torch.log(torch.abs(torch.det(J))) + logdet


# noinspection PyUnusedLocal, PyPep8Naming, PyTypeChecker
class DeepConvexFlow(torch.nn.Module):
    """
    Deep convex potential flow parameterized by an input-convex neural network. This is the main framework
        used in the paper.
    The `forward_transform_stochastic` function is used to give a stochastic estimate of the logdet "gradient"
        during training, and a stochastic estimate of the logdet itself on eval mode (using Lanczos).
    The `forward_transform_bruteforce` function computes the logdet exactly.
    """

    def __init__(self, icnn, dim, unbiased=False, no_bruteforce=True, m1=10, m2=None, rtol=0.0, atol=1e-3,
                 bias_w1=0.0, trainable_w0=True):
        super(DeepConvexFlow, self).__init__()
        if m2 is None:
            m2 = dim
        self.icnn = icnn
        self.no_bruteforce = no_bruteforce
        self.rtol = rtol
        self.atol = atol

        self.w0 = torch.nn.Parameter(torch.log(torch.exp(torch.ones(1)) - 1), requires_grad=trainable_w0)
        self.w1 = torch.nn.Parameter(torch.zeros(1) + bias_w1)
        self.bias_w1 = bias_w1

        self.m1, self.m2 = m1, m2
        self.stochastic_estimate_fn = unbiased_logdet if unbiased else \
            partial(stochastic_lanczos_quadrature, m=min(m1, dim))
        self.stochastic_grad_estimate_fn = partial(
            stochastic_logdet_gradient_estimator, m=min(m2, dim), rtol=self.rtol, atol=self.atol)

    def get_potential(self, x, context=None):
        n = x.size(0)
        if context is None:
            icnn = self.icnn(x)
        else:
            icnn = self.icnn(x, context)
        return nnF.softplus(self.w1) * icnn + nnF.softplus(self.w0) * (x.view(n, -1) ** 2).sum(1, keepdim=True) / 2

    def reverse(self, y, max_iter=1000000, lr=1.0, tol=1e-12, x=None, context=None, **kwargs):
        if x is None:
            x = y.clone().detach().requires_grad_(True)

        def closure():
            # Solves x such that f(x) - y = 0
            # <=> Solves x such that argmin_x F(x) - <x,y>
            F = self.get_potential(x, context)
            loss = torch.sum(F) - torch.sum(x * y)
            x.grad = torch.autograd.grad(loss, x)[0].detach()
            return loss

        optimizer = torch.optim.LBFGS([x], lr=lr, line_search_fn="strong_wolfe", max_iter=max_iter, tolerance_grad=tol,
                                      tolerance_change=tol)

        optimizer.step(closure)

        error_new = (self.forward_transform(x, context=context)[0] - y).abs().max().item()
        # if error_new > math.sqrt(tol):
        #     print('inversion error', error_new, flush=True)
        torch.cuda.empty_cache()
        gc.collect()

        return x

    def forward(self, x, context=None):
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            F = self.get_potential(x, context)
            f = torch.autograd.grad(F.sum(), x, create_graph=True)[0]
        return f

    def forward_transform(self, x, logdet=0, context=None, extra=None):
        if self.training or self.no_bruteforce:
            return self.forward_transform_stochastic(x, logdet, context=context, extra=extra)
        else:
            return self.forward_transform_bruteforce(x, logdet, context=context)

    def forward_transform_stochastic(self, x, logdet=0, context=None, extra=None):
        bsz, *dims = x.shape
        dim = np.prod(dims)

        with torch.enable_grad():
            x = x.clone().requires_grad_(True)

            F = self.get_potential(x, context)
            f = torch.autograd.grad(F.sum(), x, create_graph=True)[0]

            def hvp_fun(v):
                # v is (bsz, dim)
                v = v.reshape(bsz, *dims)
                hvp = torch.autograd.grad(f, x, v, create_graph=self.training, retain_graph=True)[0]

                HESS_NORM_TRACER.append((torch.norm(hvp) / torch.norm(v)).detach().cpu())

                if not torch.isnan(v).any() and torch.isnan(hvp).any():
                    raise ArithmeticError("v has no nans but hvp has nans.")
                hvp = hvp.reshape(bsz, dim)
                return hvp

        if self.training:
            v1 = sample_rademacher(bsz, dim).to(x)
            est1 = self.stochastic_grad_estimate_fn(hvp_fun, v1)
        else:
            est1 = 0

        if not self.training or (extra is not None and len(extra) > 0):
            # noinspection PyBroadException
            try:
                v2 = torch.nn.functional.normalize(sample_rademacher(bsz, dim), dim=-1).to(f)
                est2 = self.stochastic_estimate_fn(hvp_fun, v2)
            except Exception:
                import traceback
                print("stochastic_estimate_fn failed with the following error message:")
                print(traceback.format_exc(), flush=True)
                est2 = torch.zeros_like(logdet).fill_(float("nan"))
            if extra is not None and len(extra) > 0:
                extra[0] = extra[0] + est2.detach()
        else:
            est2 = 0

        return f, logdet + est1 if self.training else logdet + est2

    def forward_transform_bruteforce(self, x, logdet=0, context=None):
        warnings.warn('brute force')
        bsz = x.shape[0]
        input_shape = x.shape[1:]

        with torch.enable_grad():
            x.requires_grad_(True)
            F = self.get_potential(x, context)
            f = torch.autograd.grad(F.sum(), x, create_graph=True)[0]

            # TODO: compute Hessian in block mode instead of row-by-row.
            f = f.reshape(bsz, -1)
            H = []
            for i in range(f.shape[1]):
                retain_graph = self.training or (i < (f.shape[1] - 1))
                H.append(
                    torch.autograd.grad(f[:, i].sum(), x, create_graph=self.training, retain_graph=retain_graph)[0])

            # H is (bsz, dim, dim)
            H = torch.stack(H, dim=1)

        f = f.reshape(bsz, *input_shape)
        return f, logdet + torch.slogdet(H).logabsdet

    def extra_repr(self):
        return f"ConjGrad(rtol={self.rtol}, atol={self.atol})"


def sample_rademacher(*shape):
    return (torch.rand(*shape) > 0.5).float() * 2 - 1
