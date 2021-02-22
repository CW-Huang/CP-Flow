import numpy as np
import torch
from lib.made import MADE

# run a quick and dirty test for the autoregressive property
D = 10
rng = np.random.RandomState(14)
x = (rng.rand(1, D) > 0.5).astype(np.float32)

configs = [
    (D, [], D, False),  # test various hidden sizes
    (D, [200], D, False),
    (D, [200, 220], D, False),
    (D, [200, 220, 230], D, False),
    (D, [200, 220], D, True),  # natural ordering test
    (D, [200, 220], 2 * D, True),  # test nout > nin
    (D, [200, 220], 3 * D, False),  # test nout > nin
]

for nin, hiddens, nout, natural_ordering in configs:

    print("checking nin %d, hiddens %s, nout %d, natural %s" %
          (nin, hiddens, nout, natural_ordering))
    model = MADE(nin, hiddens, nout, natural_ordering=natural_ordering)

    # run backpropagation for each dimension to compute what other
    # dimensions it depends on.
    res = []
    for k in range(nout):
        xtr = torch.from_numpy(x).requires_grad_()
        xtrhat = model(xtr)
        loss = xtrhat[0, k]
        loss.backward()

        depends = (xtr.grad[0].numpy() != 0).astype(np.uint8)
        depends_ix = list(np.where(depends)[0])
        isok = k % nin not in depends_ix

        res.append((len(depends_ix), k, depends_ix, isok))

    # pretty print the dependencies
    res.sort()
    for nl, k, ix, isok in res:
        print("output %2d depends on inputs: %30s : %s" % (k, ix, "OK" if isok else "NOTOK"))
