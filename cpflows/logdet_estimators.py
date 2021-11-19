import numpy as np
import torch
# noinspection PyPep8Naming
import torch.nn.functional as F


EPS = 1e-7
CG_ITERS_TRACER = list()


# noinspection PyPep8Naming
def gram_schmidt_ortho(Q, v, tol=1e-5):
    """
    Orthogonalizes v wrt the rows vectors in Q.
    Assumes row vectors in Q are orthogonal and have unit Euclidean norm.

    Args:
        Q: (..., m, d)
        v: (..., d)
        tol: tolerance value for convergence
    Constraints:
        Q orthonormal, m < d.
    Returns:
        (..., d) Tensor that is orthogonal to all rows in Q.
    """
    *shape, m, d = Q.shape
    Q = Q.reshape(-1, m, d)
    assert shape == list(v.shape[:-1]), (
        f"Q and v need to have the same batch shape but"
        f" got Q.shape[:-2]={Q.shape[:-2]} and v.shape[:-1]={v.shape[:-1]}")
    v = v.reshape(-1, d)

    # Check Q is orthonormal
    with torch.no_grad():

        if m > 1:
            QQ = torch.einsum("bmd,bnd->bmn", Q, Q)
            diagmask = torch.eye(QQ.shape[-1])[None].bool()
            offdiags = QQ[~diagmask.expand_as(QQ)].reshape(-1, m, m - 1)
            if (offdiags > tol).any():
                # idx = (offdiags > tol).sum(-1)  # (b, m)
                print("Warning: non-orthogonal rows in Q")
                # print("Q", Q[idx[..., None]])

    inner_qv = torch.einsum('bmd,bd->bm', Q, v)
    proj_v = torch.einsum('bm,bmd->bd', inner_qv, Q)
    v = v - proj_v

    # EPS is added here so we can divide and multiply by the same number later.
    v_norm = torch.norm(v, dim=-1, keepdim=True).detach() + EPS

    # Verify orthogonality
    inner_qv = torch.einsum("bmd,bd->bm", Q, v / v_norm)
    retries = 0
    while (torch.abs(inner_qv) > tol).any():
        proj_v = torch.einsum('bm,bmd->bd', inner_qv * v_norm, Q)
        v = v - proj_v
        inner_qv = torch.einsum('bmd,bd->bm', Q, v)
        retries += 1
        if retries >= 10:
            print("Warning: orthogonalization exceeded 10 retries.")
            break

    return v.reshape(*shape, d)


# noinspection PyUnusedLocal,PyPep8Naming
def lanczos_tridiagonalization(hvp_fun, m, v):
    """
    Args:
        hvp_fun: A broadcastable function that computes Hessian-vector products.
        m: number of Lanczos iterations.
        v: (bsz, d) a starting orthonormal vector.
    Returns:
        A tridiagonal matrix (m, m) resulting from the Lanczos method.
    """
    bsz, d = v.shape

    # Multiple torch.stack; need better implementation.
    vecs = [v]
    Q = torch.stack(vecs, dim=1)

    w = hvp_fun(v)
    alpha = torch.einsum("bi,bi->b", w, v)
    w = gram_schmidt_ortho(Q, w)

    alphas = [alpha]
    betas = []

    for j in range(2, m + 1):
        beta = div = torch.norm(w, dim=-1)

        while (div < EPS).any():
            # TODO: only a couple batches are small; more memory efficient method?
            print(f"rerolling {(div < EPS).sum().item()} vectors")
            idx = (beta < EPS)
            w_new = torch.nn.functional.normalize(torch.randn_like(w), dim=-1)
            w_new = gram_schmidt_ortho(Q, w_new)
            w = w * ~idx.unsqueeze(-1) + w_new * idx.unsqueeze(-1)
            div = torch.norm(w, dim=-1)

        v = F.normalize(w, dim=-1)
        vecs.append(v)

        Q = torch.stack(vecs, dim=1)

        w = hvp_fun(v)
        alpha = torch.einsum("bi,bi->b", w, v)
        w = gram_schmidt_ortho(Q, w)

        alphas.append(alpha)
        betas.append(beta)

    alphas = torch.stack(alphas, dim=-1)
    betas = torch.stack(betas, dim=-1)

    T = torch.diag_embed(betas, offset=-1) + torch.diag_embed(alphas, offset=0) + torch.diag_embed(betas, offset=1)
    return T


# noinspection PyPep8Naming
def stochastic_trfA(mvp_fun, v, m):
    _, dim = v.shape
    T = lanczos_tridiagonalization(mvp_fun, m, v)
    T_eigvals, T_eigvecs = torch.symeig(T, eigenvectors=True)
    clamped_eigvals = T_eigvals.clamp(min=0).detach() + (T_eigvals - T_eigvals.detach())
    tau = T_eigvecs[..., 0, :]
    estimates = torch.sum(tau**2 * torch.log(clamped_eigvals + 1e-8), dim=-1) * dim
    return estimates


# TODO: gradient through this is very unstable.
# noinspection PyPep8Naming
def stochastic_quadrature(T, dim, func=torch.log):
    eigvals, eigvecs = torch.symeig(T, eigenvectors=True)

    with torch.no_grad():
        if eigvals.numel() > torch.unique(eigvals, dim=1).numel():
            print("Non-unique eigenvalues.")

    clamped_eigvals = eigvals.clamp(min=0).detach() + (eigvals - eigvals.detach())
    tau = eigvecs[..., 0, :]
    return torch.sum(tau * tau * func(clamped_eigvals + 1e-8), dim=-1) * dim


# noinspection PyPep8Naming
def deterministic_quadrature(T, dim):
    eigvals, _ = torch.symeig(T, eigenvectors=True)
    clamped_eigvals = eigvals.clamp(min=0).detach() + (eigvals - eigvals.detach())
    return torch.mean(torch.log(clamped_eigvals + 1e-8), dim=-1) * dim


# noinspection PyPep8Naming
def stochastic_lanczos_quadrature(hvp_fun, v, m, func=torch.log):
    bsz, dim = v.shape
    T = lanczos_tridiagonalization(hvp_fun, m, v)
    return stochastic_quadrature(T, dim, func=func)


def batch_dot_product(a, b):
    return torch.bmm(a.unsqueeze(1), b.unsqueeze(2)).squeeze(2)


# noinspection PyPep8Naming
def conjugate_gradient(hvp, b, m=10, rtol=0.0, atol=1e-3):
    """ Solves H^{-1} v using m iterations of conjugate gradient.
        v is (bsz, dim) and output shape should be (bsz, dim).
    """
    # initialization
    # could also initialize other ways, e.g. `x = torch.ones_like(b)`
    x = b.clone().detach()
    r = b - hvp(x)
    tol = atol + rtol * torch.abs(x)
    if (torch.abs(r) < tol).all():
        CG_ITERS_TRACER.append(0)
        return x
    p = r
    r2 = batch_dot_product(r, r)
    k = 0
    while k < m:
        k += 1
        Ap = hvp(p)

        a = r2 / (batch_dot_product(p, Ap) + 1e-8)
        x = x + a * p
        r = r - a * Ap
        tol = atol + rtol * torch.abs(x)
        if (torch.abs(r) < tol).all():
            break
        r2_new = batch_dot_product(r, r)
        beta = r2_new / r2
        r2 = r2_new
        p = r + beta * p
    CG_ITERS_TRACER.append(k)
    return x


# noinspection PyPep8Naming
def stochastic_logdet_gradient_estimator(hvp_fun, v, m, rtol=0.0, atol=1e-3):
    with torch.no_grad():
        v_Hinv = conjugate_gradient(hvp_fun, v, m, rtol=rtol, atol=atol)
    surrog_logdet = torch.sum(hvp_fun(v_Hinv) * v, dim=1)
    return surrog_logdet


# noinspection PyPep8Naming
def unbiased_logdet(hvp_fun, v, p=0.1, n_exact_terms=4):
    bsz, dim = v.shape
    m = geometric_sample(p) + n_exact_terms

    def coeff_fn(kk):
        return 1 / geometric_1mcdf(p, kk, n_exact_terms)
    T = lanczos_tridiagonalization(hvp_fun, m, v)
    estimate = prev_estimate = 0.0
    for k in range(n_exact_terms, m + 1):
        # logdet_estimate = deterministic_quadrature(T[:, :k, :k], dim)
        logdet_estimate = stochastic_quadrature(T[:, :k, :k], dim)
        estimate = estimate + coeff_fn(k) * (logdet_estimate - prev_estimate)
        prev_estimate = logdet_estimate
    return estimate


def geometric_sample(p):
    return np.random.geometric(p)


def geometric_1mcdf(p, k, offset=0):
    if k <= offset:
        return 1.
    else:
        k = k - offset
    """P(n >= k)"""
    return (1 - p)**max(k - 1, 0)
