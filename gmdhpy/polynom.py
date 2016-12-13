import numpy as np
import sys

from gmdhpy.gmdh_model import Model
from gmdhpy.utils import get_y_components, RefFunctionType, LayerCreationError

from sklearn import linear_model
from sklearn.cross_decomposition import PLSRegression


def set_matrix_a(ftype, u1_index, u2_index, source, source_y, a, y):
    """
    function set matrix value required to calculate polynom model coefficient
    by multiple linear regression
    """
    m = source.shape[0]
    u1x = source[:, u1_index]
    u2x = source[:, u2_index]
    for mi in range(0, m):
        # compute inputs for model
        u1 = u1x[mi]
        u2 = u2x[mi]
        a[mi, 0] = 1
        a[mi, 1] = u1
        a[mi, 2] = u2
        if RefFunctionType.rfLinearCov == ftype:
            a[mi, 3] = u1 * u2
        if RefFunctionType.rfQuadratic == ftype:
            a[mi, 3] = u1 * u2
            u1_sq = u1 * u1
            u2_sq = u2 * u2
            a[mi, 4] = u1_sq
            a[mi, 5] = u2_sq
        if RefFunctionType.rfCubic == ftype:
            a[mi, 3] = u1 * u2
            u1_sq = u1 * u1
            u2_sq = u2 * u2
            a[mi, 4] = u1_sq
            a[mi, 5] = u2_sq
            a[mi, 6] = u1_sq * u1
            a[mi, 7] = u1_sq * u2
            a[mi, 8] = u2_sq * u1
            a[mi, 9] = u2_sq * u2
    y[:] = source_y


def train_model(alpha, a, y):
    y_components = get_y_components(y)
    a2 = a[:, 1:]
    # if y_components == 1:
    if y_components > 0:
        clf = linear_model.Ridge(alpha=alpha, solver='lsqr')
        clf.fit(a2, y)
        w = np.concatenate([clf.intercept_.reshape(y_components, 1), clf.coef_], axis=1)
        # w = np.empty((len(clf.coef_[0]) + 1,), dtype=np.double)
        # w[0] = clf.intercept_[0]
        # w[1:] = clf.coef_[0]
        #
        # w = np.empty((len(clf.coef_) + 1,), dtype=np.double)
        # w[0] = clf.intercept_
        # w[1:] = clf.coef_

        return w
    elif y_components > 1:
        pls2 = PLSRegression(n_components=y_components)
        pls2.fit(a2, y)

        pass
    else:
        raise ValueError('y_components < 1')


class Polynom(object):
    def __init__(self, layer_index, u1_index, u2_index, ftype):
        self.u1_index = u1_index
        self.u2_index = u2_index
        self.ftype = ftype
        self.fw_size = 0
        self.set_type(ftype)

    def _apply(self, x):
        raise NotImplementedError

    def _transfer_linear(self, u1, u2):
        x = np.array([np.ones(u1.shape[0]), u1, u2])
        return self._apply(x)

    def _transfer_linear_cov(self, u1, u2):
        x = np.array([np.ones(u1.shape[0]), u1, u2, u1*u2])
        return self._apply(x)

    def _transfer_quadratic(self, u1, u2):
        x = np.array([np.ones(u1.shape[0]), u1, u2, u1*u2, u1*u1, u2*u2])
        return self._apply(x)

    def _transfer_cubic(self, u1, u2):
        u1_sq = u1 * u1
        u2_sq = u2 * u2
        x = np.array([np.ones(u1.shape[0]), u1, u2, u1*u2, u1_sq, u2_sq,
                      u1_sq*u1, u1_sq*u2, u2_sq*u1, u2_sq*u2])
        return self._apply(x)

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

    def transform(self, x):
        if len(x.shape) == 2:
            x1 = x[:, self.u1_index]
            x2 = x[:, self.u2_index]
        else:
            x1 = x[self.u1_index]
            x2 = x[self.u2_index]
        y = self.transfer(x1, x2)
        return y

    def fit(self, alpha, a, y):
        raise NotImplementedError


class LinearPolynom(Polynom):
    def fit(self, alpha, a, y):
        y_components = get_y_components(y)
        a2 = a[:, 1:]
        clf = linear_model.Ridge(alpha=alpha, solver='lsqr')
        clf.fit(a2, y)
        self.w = np.concatenate([clf.intercept_.reshape(y_components, 1), clf.coef_], axis=1)

    def _apply(self, x):
        return np.dot(x.T, self.w.T)


class PLSPolynom(Polynom):
    def fit(self, alpha, a, y):
        y_components = get_y_components(y)
        a2 = a[:, 1:]
        self.pls = PLSRegression(n_components=y_components)
        self.pls.fit(a2, y)

    def _apply(self, x):
        y = self.pls.transform(x.T[:, 1:])
        return y

class PolynomModel(Model):
    """Polynomial GMDH model class
    """

    def __init__(self, gmdh, layer_index, u1_index, u2_index, ftype, model_index, model_type):
        super(PolynomModel, self).__init__(gmdh, layer_index, u1_index, u2_index, model_index)
        if model_type.lower() == 'polynom':
            self.f = LinearPolynom(layer_index, u1_index, u2_index, ftype)
            self.fm = LinearPolynom(layer_index, u1_index, u2_index, ftype)
        elif model_type.lower() == 'pls':
            self.f = PLSPolynom(layer_index, u1_index, u2_index, ftype)
            self.fm = PLSPolynom(layer_index, u1_index, u2_index, ftype)
        else:
            raise NotImplementedError

    def get_name(self):
        if self.f.ftype == RefFunctionType.rfLinear:
            return 'w0 + w1*xi + w2*xj'
        elif self.f.ftype == RefFunctionType.rfLinearCov:
            return 'w0 + w1*xi + w2*xj + w3*xi*xj'
        elif self.f.ftype == RefFunctionType.rfQuadratic:
            return 'full polynom 2nd degree'
        elif self.f.ftype == RefFunctionType.rfCubic:
            return 'full polynom 3rd degree'
        else:
            return 'Unknown'

    def get_short_name(self):
        if self.f.ftype == RefFunctionType.rfLinear:
            return 'linear'
        elif self.f.ftype == RefFunctionType.rfLinearCov:
            return 'linear cov'
        elif self.f.ftype == RefFunctionType.rfQuadratic:
            return 'quadratic'
        elif self.f.ftype == RefFunctionType.rfCubic:
            return 'cubic'
        else:
            return 'Unknown'

    def __repr__(self):
        s = 'PolynomModel {0} - {1}\n'.format(self.model_index, RefFunctionType.get_name(self.ref_function_type))
        s += 'u1: {0}\n'.format(self.get_features_name(self.u1_index))
        s += 'u2: {0}\n'.format(self.get_features_name(self.u2_index))
        s += 'train error: {0}\n'.format(self.train_err)
        s += 'validate error: {0}\n'.format(self.validate_err)
        s += 'bias error: {0}\n'.format(self.bias_err)
        for n in range(0, self.f.w.shape[0]):
            s += 'w{0}={1}'.format(n, self.f.w[n])
            if n < self.f.w.shape[0] - 1:
                s += '; '
            else:
                s += '\n'
        s += '||w||^2={ww}'.format(ww=self.f.w.mean())
        return s

    def _calculate_model_weights(self, train_x, train_y, validate_x, validate_y, alpha):
        """
        Calculate model coefficients
        """

        n_train = train_x.shape[0]
        n_validate = validate_x.shape[0]
        y_components = get_y_components(train_y)
        # declare Y variables of multiple linear regression for train and test targets
        ya = np.empty((n_train, y_components), dtype=np.double)
        yb = np.empty((n_validate, y_components), dtype=np.double)
        # ya = np.empty(n_train, dtype=np.double)
        # yb = np.empty(n_validate, dtype=np.double)
        min_rank = 2
        rank_b = 10000000

        # declare X variables of multiple linear regression for train and test targets
        a = np.empty((n_train, self.f.fw_size), dtype=np.double)
        b = np.empty((n_validate, self.f.fw_size), dtype=np.double)

        self.train_err = sys.float_info.max
        self.validate_err = sys.float_info.max

        # set X and Y variables of multiple linear regression
        # Train the model using train and test sets
        try:
            set_matrix_a(self.f.ftype, self.u1_index, self.u2_index, train_x, train_y, a, ya)
            self.f.fit(alpha, a, ya)
            rank_a = 10
            pass
        except:
            raise LayerCreationError('Error training model on train data set', self.layer_index)

        if self.need_bias_stuff():
            try:
                set_matrix_a(self.fm.ftype, self.u1_index, self.u2_index, validate_x, validate_y, b, yb)
                self.fm.fit(alpha, b, yb)
                rank_b = 10
            except:
                raise LayerCreationError('Error training model on train data set', self.layer_index)


        # train_y = train_y[:, 0]
        # validate_y = validate_y[:, 0]

        self.bias_err = 0
        if rank_a < min_rank or rank_b < min_rank:
            self.valid = False
        else:
            self.valid = True
            # calculate model errors
            if self.need_bias_stuff():
                self.bias_err = self.get_bias_err(train_x, validate_x, train_y, validate_y)

            self.train_err = self.get_regularity_err(train_x, train_y)
            self.validate_err = self.get_regularity_err(validate_x, validate_y)

    def fit(self, index, train_x, train_y, validate_x, validate_y, **fit_params):
        self.fit_params = fit_params
        self._calculate_model_weights(
            train_x,
            train_y,
            validate_x,
            validate_y,
            fit_params['alpha'])


    def _refit(self, data_x, data_y):
        """
        Fit model on total (original) data set (train and validate sets)
        """
        data_len = data_x.shape[0]
        a = np.empty((data_len, self.f.fw_size), dtype=np.double)
        y = np.empty(data_len, dtype=np.double)
        # Fit the model using all data

        try:
            set_matrix_a(self.f.ftype, self.u1_index, self.u2_index, data_x, data_y, a, y)
            self.f.w = train_model(self.fit_params['alpha'], a, y)
            pass
        except:
            raise LayerCreationError('Error training model on train data set', self.layer_index)





