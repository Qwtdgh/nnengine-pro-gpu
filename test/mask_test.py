import os.path
import pickle

import pytest

from lightGE.core import Tensor, Transformer
import numpy as np


class TestMask:

    def test_mask(self):
        cached_path = '../tmp/test_mask.pkl'
        batch = 1
        sentence_len = 10
        n_inputs = 4
        inputLen = [4]
        outputLen = [5]
        batchInput1 = Tensor(np.array([[[-2.49739425e-01, -7.60393613e-01, -6.36791672e-01, -1.00987416e+00],
                                        [5.17415169e-01, 1.31487395e+00, 3.23991877e-01, -1.19690298e-03],
                                        [7.01543295e-01, 3.67174903e-01, -8.84372202e-01, 7.85204927e-01],
                                        [1.37859910e+00, 1.23994663e+00, 2.32805357e-01, -1.57464240e-01],
                                        [8.38917660e-01, -1.25990406e+00, 2.54823951e-02, 6.31936887e-01],
                                        [8.11388273e-01, 8.17781365e-03, -7.10074510e-01, -6.39614111e-01],
                                        [-1.76559083e+00, -7.88812595e-03, 2.87746582e-02, 9.98895095e-01],
                                        [6.49503467e-01, -4.04000604e-01, 1.15786584e+00, -5.41454961e-01],
                                        [-4.08259077e-01, -2.02076424e-01, -1.54162652e+00, 3.46309891e-01],
                                        [1.45335076e+00, 1.48273808e+00, -1.77095812e+00, -6.49452065e-01]]]))
        batchOutput1 = Tensor(np.array([[[1.63524569, -0.51189901, -0.53163936, 0.29592387],
                                         [-0.63365053, -2.46431477, -0.29354211, -1.86577737],
                                         [-0.16473815, -0.47106771, 1.39843749, -0.14246705],
                                         [1.83140876, -0.15853078, 0.40527597, -0.16813534],
                                         [-0.74466711, -0.69254871, 0.48946672, -0.52769667],
                                         [-1.25991781, -1.14927568, 0.28552546, -0.55574188],
                                         [-0.15123766, 0.06704581, 0.39418653, 1.41817566],
                                         [-0.66283581, -1.40019203, -0.57248706, -0.672118],
                                         [0.44024666, 1.76037769, 0.81913022, 2.09428447],
                                         [-0.15162112, 0.41419923, 0.56462677, 1.40333091]]]))
        batchInput2 = Tensor(np.concatenate((batchInput1.data[:, :inputLen[0], :], np.array(
            [[[100, 100, 100, 100] for i in range(sentence_len - inputLen[0])]])), axis=1))
        batchOutput2 = Tensor(np.concatenate((batchOutput1.data[:, :outputLen[0], :], np.array(
            [[[100, 100, 100, 100] for i in range(sentence_len - outputLen[0])]])), axis=1))

        if os.path.exists(cached_path):
            [T] = pickle.load(open(cached_path, 'rb'))
        else:
            T = Transformer(encoder_block=5, decoder_block=1, batch=batch, sentence_len=sentence_len, n_inputs=n_inputs,
                            drop_prob=0.1, vocab_len=10)
            pickle.dump([T], open(cached_path, 'wb'))

        y1: Tensor = T(tuple((batchInput1, batchOutput1)), lens=tuple((inputLen, outputLen)))
        y2: Tensor = T(tuple((batchInput2, batchOutput2)), lens=tuple((inputLen, outputLen)))
        assert (y1.data[:, :outputLen[0], :] == y2.data[:, :outputLen[0], :]).all()


if __name__ == '__main__':
    pytest.main()
