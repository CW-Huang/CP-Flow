from sklearn.datasets import make_swiss_roll
import torch
import numpy as np
from torch.utils.data import Dataset as Dataset
from lib.distributions import log_normal
from scipy.stats import wishart
from sklearn.utils import shuffle as util_shuffle


class ToyDataset(Dataset):
    data = dict()
    data_names = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.data[cls.__name__] = cls
        cls.data_names.append(cls.__name__)

    def __init__(self, n=50000):
        self.data = self.sample(n)

    def sample(self, batch_size):
        raise NotImplementedError

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class SwissRoll(ToyDataset):
    """
    Swiss roll distribution sampler.
    noise control the amount of noise injected to make a thicker swiss roll
    """
    def sample(self, batch_size, noise=0.5):
        return torch.from_numpy(
            make_swiss_roll(batch_size, noise)[0][:, [0, 2]].astype('float64') / 5.)


# taken from https://github.com/nicola-decao/BNAF/blob/master/data/generate2d.py
class TwoSpirals(ToyDataset):
    def sample(self, batch_size):
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        return torch.from_numpy(x).float()


class EightGaussian(ToyDataset):
    def sample(self, batch_size, return_idx=False):
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        indices = []
        for i in range(batch_size):
            point = np.random.randn(2) * 0.5
            idx = np.random.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
            indices.append(idx)
        dataset = np.array(dataset, dtype='float32')
        dataset /= 1.414
        if return_idx:
            return torch.from_numpy(dataset), torch.from_numpy(np.array(indices))
        else:
            return torch.from_numpy(dataset)


class Rings(ToyDataset):
    def sample(self, batch_size):
        n_samples4 = n_samples3 = n_samples2 = batch_size // 4
        n_samples1 = batch_size - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25

        X = np.vstack([
            np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
            np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
        ]).T * 3.0
        X = util_shuffle(X)

        # Add noise
        X = X + np.random.normal(scale=0.08, size=X.shape)

        return torch.from_numpy(X.astype("float32"))


class MAFMoon(ToyDataset):
    def sample(self, batch_size):
        x = torch.randn(batch_size, 2)
        x[:,0] += x[:,1] ** 2
        x[:,0] /= 2
        x[:,0] -= 2
        return x


def gaussian_sampler(mean, log_var):
    sigma = torch.exp(0.5*log_var)
    return torch.randn(*mean.shape) * sigma + mean


class OneDMixtureOfGaussians(ToyDataset):
    """
    Mixture of Gaussians:
        x ~ p(x; pi) = pi N(m1, var) + (1-pi) N(m2, var)
        with pi ~ uniform(0,1)
    """

    m1 = 2 * torch.ones(1)
    m2 = -2 * torch.ones(1)
    log_var = -1 * torch.ones(1)

    def sample(self, batch_size):
        m1 = torch.ones(batch_size, 1) * self.m1
        m2 = torch.ones(batch_size, 1) * self.m2
        log_var = torch.ones(batch_size, 1) * self.log_var
        s1 = gaussian_sampler(m1, log_var)
        s2 = gaussian_sampler(m2, log_var)
        pi = torch.rand(batch_size, 1)
        mask = torch.le(torch.rand(batch_size, 1), pi).float()
        return torch.cat([mask * s1 + (1-mask) * s2, pi], 1)

    def logp(self, x, pi):
        log_pi1 = np.log(pi)
        log_pi2 = np.log(1-pi)
        lp1 = log_normal(x, self.m1, self.log_var)
        lp2 = log_normal(x, self.m2, self.log_var)
        return torch.logsumexp(torch.cat([lp1+log_pi1, lp2+log_pi2], 1), 1)

    def plot_logp(self):
        import matplotlib.pyplot as plt
        xx = torch.linspace(-5, 5, 1000).unsqueeze(1)
        for pi in np.linspace(0.1, 0.9, 11):
            p = torch.exp(self.logp(xx, pi)).data.numpy()
            plt.plot(xx, p)


class Gaussian(ToyDataset):

    def __init__(self, n, dim=2, m=None, C=None):
        if m is None:
            m = np.random.randn(dim)
        if C is None:
            C = wishart.rvs(dim+1, np.identity(dim), 1)
        self.m = m
        self.C = C
        super(Gaussian, self).__init__(n)

    def sample(self, batch_size):
        return torch.from_numpy(np.random.multivariate_normal(self.m, self.C, batch_size)).float()

    @property
    def entropy(self):
        return 0.5 * np.linalg.slogdet(2*np.pi*np.exp(1)*self.C)[1]
