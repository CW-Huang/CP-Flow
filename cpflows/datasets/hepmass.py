"""
Copyright (c) 2017, George Papamakarios
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of anybody else.
"""
import pandas as pd
import numpy as np
from collections import Counter
from os.path import join


class HEPMASS:
    """
    The HEPMASS data set.
    http://archive.ics.uci.edu/ml/datasets/HEPMASS
    """

    class Data:

        def __init__(self, data):

            self.x = data.astype(np.float32)
            self.N = self.x.shape[0]

    def __init__(self, root):

        path = join(root, 'hepmass/')
        trn, val, tst = load_data_no_discrete_normalised_as_array(path)

        self.trn = self.Data(trn)
        self.val = self.Data(val)
        self.tst = self.Data(tst)

        self.n_dims = self.trn.x.shape[1]


def load_data(path):

    data_train = pd.read_csv(filepath_or_buffer=join(path, "1000_train.csv"), index_col=False)
    data_test = pd.read_csv(filepath_or_buffer=join(path, "1000_test.csv"), index_col=False)

    return data_train, data_test


def load_data_no_discrete(path):
    """
    Loads the positive class examples from the first 10 percent of the dataset.
    """
    data_train, data_test = load_data(path)

    # Gets rid of any background noise examples i.e. class label 0.
    data_train = data_train[data_train[data_train.columns[0]] == 1]
    data_train = data_train.drop(data_train.columns[0], axis=1)
    data_test = data_test[data_test[data_test.columns[0]] == 1]
    data_test = data_test.drop(data_test.columns[0], axis=1)
    # Because the data set is messed up!
    data_test = data_test.drop(data_test.columns[-1], axis=1)

    return data_train, data_test


def load_data_no_discrete_normalised(path):

    data_train, data_test = load_data_no_discrete(path)
    mu = data_train.mean()
    s = data_train.std()
    data_train = (data_train - mu) / s
    data_test = (data_test - mu) / s

    return data_train, data_test


def load_data_no_discrete_normalised_as_array(path):

    data_train, data_test = load_data_no_discrete_normalised(path)
    data_train, data_test = data_train.as_matrix(), data_test.as_matrix()

    i = 0
    # Remove any features that have too many re-occurring real values.
    features_to_remove = []
    for feature in data_train.T:
        c = Counter(feature)
        max_count = np.array([v for k, v in sorted(c.items())])[0]
        if max_count > 5:
            features_to_remove.append(i)
        i += 1
    data_train = data_train[:, np.array([i for i in range(data_train.shape[1]) if i not in features_to_remove])]
    data_test = data_test[:, np.array([i for i in range(data_test.shape[1]) if i not in features_to_remove])]

    N = data_train.shape[0]
    N_validate = int(N * 0.1)
    data_validate = data_train[-N_validate:]
    data_train = data_train[0:-N_validate]

    return data_train, data_validate, data_test
