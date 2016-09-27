import sys
from gmdhpy.polynom import PolynomModel

#***********************************************************************************************************************
#   GMDH layer
#***********************************************************************************************************************
class LayerCreationError(Exception):
    """raised when error happens while layer creation
    """
    def __init__(self, message, layer_index):
        # Call the base class constructor with the parameters it needs
        super(LayerCreationError, self).__init__(message)
        self.layer_index = layer_index


class Layer(list):
    """Layer class of multilayered group method of data handling algorithm
    """

    def __init__(self, gmdh, layer_index, *args):
        list.__init__(self, *args)
        self.layer_index = layer_index
        self.l_count = gmdh.l_count
        self.n_features = gmdh.n_features
        self.err = sys.float_info.max
        self.train_err = sys.float_info.max
        self.valid = True
        self.input_index_set = set([])

    def add_polynomial_model(self, gmdh, index_u1, index_u2, ftype):
        """Add polynomial model to the layer
        """
        self.add(PolynomModel(gmdh, self.layer_index, index_u1, index_u2, ftype, len(self)))

    def __repr__(self):
        st = '*********************************************\n'
        s = st
        s += 'Layer {0}\n'.format(self.layer_index)
        s += st
        for n, model in enumerate(self):
            s += '\n'
            s += model.__repr__()
            if n == len(self) - 1:
                s += '\n'
        return s

    def add(self, model):
        model.model_index = len(self)
        self.append(model)
        self.input_index_set.add(model.u1_index)
        self.input_index_set.add(model.u2_index)

    def delete(self, index):
        self.pop(index)
        for n in range(index, len(self)):
            self[n].model_index = n
        self.input_index_set.clear()
        for model in self:
            self.input_index_set.add(model.u1_index)
            self.input_index_set.add(model.u2_index)
