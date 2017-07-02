# -*- coding: utf-8 -*-
from __future__ import print_function
import graphviz as gv
import functools
graph = functools.partial(gv.Graph, format='svg')
digraph = functools.partial(gv.Digraph, format='svg')
# http://matthiaseisen.com/articles/graphviz/


class PlotModel:
    """Plot self-organising polynomial neural network (multilayered GMDH)
    """
    def __init__(self, model, filename, plot_neuron_name=False, view=False):
        self.g = gv.Digraph(format='svg')
        self.output = 'OUTPUT'
        self.model = model
        self.plot_neuron_name = plot_neuron_name
        self.filename = filename
        self.view = view

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
        self.g.graph_attr.update(label='Self-organizing deep learning polynomial neural network\n ', labelloc='t', center='true',
                                 fontsize='18')

    def _get_feature_name(self, index):
        s = 'F{0}'.format(index)
        if self.model.feature_names is not None and len(self.model.feature_names) > 0:
            s += '\n{0}'.format(self.model.feature_names[index])
        return s

    def _get_neuron_name(self, neuron):
        s = 'layer {0}\nneuron {1}'.format(neuron.layer_index, neuron.neuron_index, neuron.get_name())
        if self.plot_neuron_name:
            s += '\n{0}'.format(neuron.get_short_name())
        return s

    @staticmethod
    def _get_feature_index(layers, neuron, u_index):
        if neuron.layer_index == 0:
            return True, u_index
        else:
            prev_layer = layers[neuron.layer_index-1]
            if u_index < len(prev_layer):
                return False, u_index
            else:
                return True, u_index - len(prev_layer)

    def _add_connection(self, layers, neuron, u_index):
        input_is_original_feature, feature_index = self._get_feature_index(layers, neuron, u_index)
        if input_is_original_feature:
            return self._add_edge(self._get_feature_name(feature_index), self._get_neuron_name(neuron))
        else:
            prev_layer = layers[neuron.layer_index-1]
            parent_neuron = prev_layer[feature_index]
            return self._add_edge(self._get_neuron_name(parent_neuron), self._get_neuron_name(neuron))

    def _add_edge(self, a, b):
        return self.g.edge(a, b, color=self.connection_color, fillcolor=self.connection_fill_color, weight='1')

    def plot(self):
        if len(self.model.layers) == 0:
            return

        features_graph = digraph()
        for i in range(0, self.model.n_features):
            features_graph.node(self._get_feature_name(i), **self.io_node_param)
        self.g.subgraph(features_graph)

        for layer in self.model.layers:
            layer_graph = digraph()
            for neuron in layer:
                s = self._get_neuron_name(neuron)
                layer_graph.node(s, **self.node_param)
            self.g.subgraph(layer_graph)

        for layer in self.model.layers:
            for neuron in layer:
                self._add_connection(self.model.layers, neuron, neuron.u1_index)
                self._add_connection(self.model.layers, neuron, neuron.u2_index)

        last_layer = self.model.layers[-1]
        last_neuron = last_layer[0]
        self._add_edge(self._get_neuron_name(last_neuron), self.output)

        self.g.render(filename=self.filename, view=self.view)
