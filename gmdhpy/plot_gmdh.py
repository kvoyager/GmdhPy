__author__ = 'Konstantin Kolokolov'

import graphviz as gv
import functools
graph = functools.partial(gv.Graph, format='svg')
digraph = functools.partial(gv.Digraph, format='svg')

# http://matthiaseisen.com/articles/graphviz/


class PlotGMDH:
    """

    """
    def __init__(self, gmdh, filename, plot_model_name=False, view=False):
        self.g = gv.Digraph(format='svg')
        self.output = 'OUTPUT'
        self.gmdh = gmdh
        self.plot_model_name = plot_model_name

        '''
        # custom palette
        self.node_color = '#ccd1ff'
        self.io_node_color = '#bed4eb'
        self.io_pen_color = '#535ebe'
        self.pen_color = '#535ebe'
        self.io_font_color = 'black'
        self.connection_color = '#ccd1ff'
        self.connection_fill_color = '#535ebe'
        '''

        # scikit-learn palette
        self.node_color = '#cde8ef'
        self.io_node_color = '#ff9c34'
        self.pen_color = '#cde8ef'
        self.io_pen_color = '#f89939'
        self.io_font_color = 'white'
        self.connection_color = '#cde8ef'
        self.connection_fill_color = '#3499cd'

        self.io_node_param = ({'style': 'filled', 'color': self.io_pen_color, 'fillcolor': self.io_node_color,
                               'fontsize': '12', 'fontcolor': self.io_font_color, 'rank': 'same'})
        self.node_param = ({'style': 'filled', 'shape': 'rect', 'color': self.pen_color, 'fillcolor': self.node_color,
                               'fontsize': '10'})
        self.g.node(self.output, **self.io_node_param)
        self.g.graph_attr.update(label='Multilayered group method of data handling model\n ', labelloc='t', center='true',
                                 fontsize='18')
        self.plot_gmdh(gmdh, filename, view)

    def get_feature_name(self, index):
        s = 'F{0}'.format(index)
        if self.gmdh.feature_names is not None and len(self.gmdh.feature_names) > 0:
            s += '\n{0}'.format(self.gmdh.feature_names[index])
        return s

    def get_model_name(self, model):
        s = 'layer {0}\nmodel {1}'.format(model.layer_index, model.model_index, model.get_name())
        if self.plot_model_name:
            s += '\n{0}'.format(model.get_short_name())
        return s

    @classmethod
    def get_feature_index(cls, model, u_index):
        if model.layer_index == 0:
            return True, u_index
        else:
            prev_layer = model.layers[model.layer_index-1]
            if u_index < len(prev_layer):
                return False, u_index
            else:
                return True, u_index - len(prev_layer)

    def add_connection(self, model, u_index):
        input_is_original_feature, feature_index = self.get_feature_index(model, u_index)
        if input_is_original_feature:
            return self.add_edge(self.get_feature_name(feature_index), self.get_model_name(model))
        else:
            prev_layer = model.layers[model.layer_index-1]
            parent_model = prev_layer[feature_index]
            return self.add_edge(self.get_model_name(parent_model), self.get_model_name(model))

    def add_edge(self, a, b):
        return self.g.edge(a, b, color=self.connection_color, fillcolor=self.connection_fill_color, weight='1')

    def plot_gmdh(self, gmdh, filename, view):
        if len(gmdh.layers) == 0:
            return

        features_graph = digraph()
        #features_graph = gv.graph(self.g, features_graph)
        for i in range(0, gmdh.n_features):
            features_graph.node(self.get_feature_name(i), **self.io_node_param)
        self.g.subgraph(features_graph)

        for layer in gmdh.layers:
            layer_graph = digraph()
            for model in layer:
                s = self.get_model_name(model)
                layer_graph.node(s, **self.node_param)
            self.g.subgraph(layer_graph)

        for n, layer in enumerate(gmdh.layers):
            for model in layer:
                self.add_connection(model, model.u1_index)
                self.add_connection(model, model.u2_index)

        last_layer = gmdh.layers[len(gmdh.layers)-1]
        last_model = last_layer[0]
        self.add_edge(self.get_model_name(last_model), self.output)

        self.g.render(filename=filename, view=view)
