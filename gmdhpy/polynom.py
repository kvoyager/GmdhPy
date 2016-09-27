import numpy as np

from gmdhpy.gmdh_model import Model, RefFunctionType


class PolynomModel(Model):
    """Polynomial GMDH model class
    """

    def __init__(self, gmdh, layer_index, u1_index, u2_index, ftype, model_index):
        super(PolynomModel, self).__init__(gmdh, layer_index, u1_index, u2_index, model_index)
        self.ftype = ftype
        self.fw_size = 0
        self.set_type(ftype)
        self.w = np.array([self.fw_size], dtype=np.double)
        self.wt = np.array([self.fw_size], dtype=np.double)

    def copy_result(self, source):
        self.w = np.copy(source[1])
        self.wt = np.copy(source[2])
        self.valid = source[3]
        self.bias_err = source[4]
        self.train_err = source[5]
        self.test_err = source[6]


    def _transfer_linear(self, u1, u2, w):
        return w[0] + w[1]*u1 + w[2]*u2

    def _transfer_linear_cov(self, u1, u2, w):
        return w[0] + u1*(w[1] + w[3]*u2) + w[2]*u2

    def _transfer_quadratic(self, u1, u2, w):
        return w[0] + u1*(w[1] + w[3]*u2 + w[4]*u1) + u2*(w[2] + w[5]*u2)

    def _transfer_cubic(self, u1, u2, w):
        u1_sq = u1*u1
        u2_sq = u2*u2
        return w[0] + w[1]*u1 + w[2]*u2 + w[3]*u1*u2 + w[4]*u1_sq + w[5]*u2_sq + \
            w[6]*u1*u1_sq + w[7]*u1_sq*u2 + w[8]*u1*u2_sq + w[9]*u2*u2_sq

    def set_type(self, new_type):
        self.ref_function_type = new_type
        if new_type == RefFunctionType.rfLinear:
            self.transfer = self._transfer_linear
            self.fw_size = 3
        elif new_type == RefFunctionType.rfLinearCov:
            self.transfer = self._transfer_linear_cov
            self.fw_size = 4
        elif new_type == RefFunctionType.rfQuadratic:
            self.transfer = self._transfer_quadratic
            self.fw_size = 6
        elif new_type == RefFunctionType.rfCubic:
            self.transfer = self._transfer_cubic
            self.fw_size = 10
        else:
            raise NotImplementedError

    @staticmethod
    def get_regularity_err(u1_index, u2_index, transfer, x, y, w):
        """Calculation of regularity error
        """
        data_len = x.shape[0]
        x1 = x[:, u1_index]
        x2 = x[:, u2_index]
        yt = np.empty((data_len,), dtype=np.double)

        for m in range(0, data_len):
            yt[m] = transfer(x1[m], x2[m], w)

        s = ((y - yt) ** 2).sum()
        s2 = (y ** 2).sum()
        err = s/s2
        return err

    @staticmethod
    def get_sub_bias_err(u1_index, u2_index, transfer, x, len, w, wt):
        """Helper function for calculation of unbiased error
        """
        x1 = x[:, u1_index]
        x2 = x[:, u2_index]
        yta = np.empty((len,), dtype=np.double)
        ytb = np.empty((len,), dtype=np.double)

        for m in range(0, len):
            yta[m] = transfer(x1[m], x2[m], w)
            ytb[m] = transfer(x1[m], x2[m], wt)

        s = ((yta - ytb) ** 2).sum()
        return s

    @staticmethod
    def get_bias_err(u1_index, u2_index, transfer, train_x, test_x, train_y, test_y, w, wt):
        """Calculation of unbiased error
        """
        s = 0
        n_train = train_x.shape[0]
        n_test = test_x.shape[0]
        s += PolynomModel.get_sub_bias_err(u1_index, u2_index, transfer, train_x, n_train, w, wt)
        s += PolynomModel.get_sub_bias_err(u1_index, u2_index, transfer, test_x, n_test, w, wt)
        s2 = (train_y ** 2).sum() + (test_y ** 2).sum()
        err = s/s2
        return err

    def get_name(self):
        if self.ftype == RefFunctionType.rfLinear:
            return 'w0 + w1*xi + w2*xj'
        elif self.ftype == RefFunctionType.rfLinearCov:
            return 'w0 + w1*xi + w2*xj + w3*xi*xj'
        elif self.ftype == RefFunctionType.rfQuadratic:
            return 'full polynom 2nd degree'
        elif self.ftype == RefFunctionType.rfCubic:
            return 'full polynom 3rd degree'
        else:
            return 'Unknown'

    def get_short_name(self):
        if self.ftype == RefFunctionType.rfLinear:
            return 'linear'
        elif self.ftype == RefFunctionType.rfLinearCov:
            return 'linear cov'
        elif self.ftype == RefFunctionType.rfQuadratic:
            return 'quadratic'
        elif self.ftype == RefFunctionType.rfCubic:
            return 'cubic'
        else:
            return 'Unknown'

    def __repr__(self):
        s = 'PolynomModel {0} - {1}\n'.format(self.model_index, RefFunctionType.get_name(self.ref_function_type))
        s += 'u1: {0}\n'.format(self.get_features_name(self.u1_index))
        s += 'u2: {0}\n'.format(self.get_features_name(self.u2_index))
        s += 'train error: {0}\n'.format(self.train_err)
        s += 'test error: {0}\n'.format(self.test_err)
        s += 'bias error: {0}\n'.format(self.bias_err)
        for n in range(0, self.w.shape[0]):
            s += 'w{0}={1}'.format(n, self.w[n])
            if n < self.w.shape[0] - 1:
                s += '; '
            else:
                s += '\n'
        s += '||w||^2={ww}'.format(ww=self.w.mean())
        return s

    def fit(self):
        raise NotImplementedError
