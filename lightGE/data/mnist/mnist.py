import gzip

import numpy as np

from lightGE.data import Dataset


class MnistDataset(Dataset):
    def __init__(self):
        super(MnistDataset, self).__init__()

    def load_data(self, data_dir):
        def extract_data(filename, num_data, head_size, data_size):
            with gzip.open(filename) as bytestream:
                bytestream.read(head_size)
                buf = bytestream.read(data_size * num_data)
                data = np.frombuffer(buf, dtype=np.uint8).astype(np.float64)
            return data

        data = extract_data(data_dir + 'train-images-idx3-ubyte.gz', 60000, 16, 28 * 28)
        trX = data.reshape((60000, 28, 28, 1))

        data = extract_data(data_dir + 'train-labels-idx1-ubyte.gz', 60000, 8, 1)
        trY = data.reshape((60000))

        data = extract_data(data_dir + 't10k-images-idx3-ubyte.gz', 10000, 16, 28 * 28)
        teX = data.reshape((10000, 28, 28, 1))

        data = extract_data(data_dir + 't10k-labels-idx1-ubyte.gz', 10000, 8, 1)
        teY = data.reshape((10000))

        trY = np.asarray(trY)
        teY = np.asarray(teY)

        x = np.concatenate((trX, teX), axis=0)
        y = np.concatenate((trY, teY), axis=0).astype(np.int32)

        data_index = np.arange(x.shape[0])
        np.random.shuffle(data_index)
        # data_index = data_index[:128]
        x = x[data_index, :, :, :]
        y = y[data_index]
        y_vec = np.zeros((len(y), 10), dtype=np.float64)
        for i, label in enumerate(y):
            y_vec[i, y[i]] = 1.0

        x /= 255.
        x = x.transpose(0, 3, 1, 2)

        self.x = x
        self.y = y_vec