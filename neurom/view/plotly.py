'''
Define the public 'draw' function to be used to draw
morphology using plotly
'''
from __future__ import \
    absolute_import  # prevents name clash with local plotly module
from itertools import chain, cycle, islice, repeat
from collections import defaultdict
import six
import copy

import numpy as np
from neurom import NeuriteType

try:
    from plotly.matplotlylib import mpltools
    import plotly.graph_objs as go
    from plotly.offline import plot, iplot, init_notebook_mode
except ImportError:
    raise ImportError(
        'neurom[plotly] is not installed. Please install it by doing: pip install neurom[plotly]')

from neurom import COLS, iter_segments, iter_neurites, iter_sections
from neurom.view.view import TREE_COLOR

PCA_AXIS_INFOS = [('blue', '3rd pca axis'), ('green', '2sd pca axis'), ('red', '1st pca axis')]
NEURON_NAME = 'neuron'
NEURON_ROTATED_NAME = 'neuron rotated'
SOMA_NAME = 'soma'
PCA_SUFFIX = 'pca'
CENTERED_SUFFIX = 'centered'
ROTATION_PREFIX = 'Rotated'


def draw(obj, plane='3d', inline=False, **kwargs):
    '''Draw the morphology within the given plane

    plane (str): a string representing the 2D plane (example: 'xy')
                 or '3d', '3D' for a 3D view

    inline (bool): must be set to True for interactive ipython notebook plotting
    '''
    title = 'neuron'
    return _plot_neuron(obj, plane, title, inline, **kwargs)


def draw_pca(pcas, plane='3d', global_rotation=None, inline=False, **kwargs):
    ''' Draw a morphology, pcas related to neurites and a rotated version of the morphology

    The plot correspond to a 3/2d version of the morphology. Pcas are computed on group of neurites
    and displayed on the figure. A rotated version of the morphology (with neurites aligned on
    their preferred axis) is also displayed.

        Args:
            pcas (PCA or MultiPCA): the pcas you want to draw
            plane (str): a string representing the 2D plane (example: 'xy')
                         or '3d', '3D' for a 3D view
            global_rotation (PCA or dict) force the rotation to be global for the whole neuron.
                        The rotation is then defined only by the global rotation pca.
            inline (bool): must be set to True for interactive ipython notebook plotting

        Notes :
            If the global rotation is defined then all pcas from pcas are plotted but only
            the one defined in global rotation is used to rotate the neuron
    '''
    pcas = _sanitize_pca(pcas)
    title = 'neuron-pca'
    return _plot_pca(pcas, plane, title, global_rotation, inline, **kwargs)


def draw_neuron_compare(neuron1, neuron2, name1='neuron1', name2='neuron2', color1='blue',
                        color2='red', plane='3d', inline=False, draw=True, **kwargs):
    '''Draw the two morphologies within the given plane

        Args:
            neuron1 (morph) : first neuron morphology
            neuron2 (morph) : second neuron morphology
            name1 (str) : name of the first morphology
            name2 (str) : name of the second morphology
            color1 (str): color of the first morphology can be a rgb string or hex or css name
            color2 (str): color of the second morphology can be a rgb string or hex or css name
            plane (str): a string representing the 2D plane (example: 'xy')
                         or '3d', '3D' for a 3D view
            inline (bool): must be set to True for interactive ipython notebook plotting
        '''
    title = 'compare-{}-{}'.format(name1, name2)
    return _plot_compare_neurons(neuron1, name1, color1, neuron2, name2, color2, plane, title,
                                 inline, draw, **kwargs)


def draw_pca_compare(neuron_ref, pcas, plane='3d', global_rotation=None, inline=False, **kwargs):
    '''Draw a morphology and its pca within the given plane

        Args:
            neuron_ref (morph) : neuron morphology of reference
            pcas (MultiPCA or PCA): pcas containing information for neurite rotations
            plane (str): a string representing the 2D plane (example: 'xy')
                         or '3d', '3D' for a 3D view
            global_rotation (PCA): optional PCA applied to the whole neuron
            inline (boolean): must be set to True for interactive ipython notebook plotting

        Notes :
            If the global rotation is defined then all pcas from pcas are not used and only
            the one defined in global rotation is used to rotate the neuron
        '''
    pcas = _sanitize_pca(pcas)
    return _plot_pca_compare(neuron_ref, pcas, plane, global_rotation, inline, True, **kwargs)


def _sanitize_pca(pcas):
    ''' Sanitize the pcas. Check if isinstance of PCA or MultiPCA or raise.
            Args:
                neuron_ref (morph) : neuron morphology of reference
                pcas (MultiPCA or PCA): pcas containing information for neurite rotations
            Returns:
              pcas : a MultiPCA
    '''
    if isinstance(pcas, PCA):
        return MultiPCA(pcas.neuron, {'name': pcas.name, 'group': pcas.group, 'axis': pcas.axis})
    elif not isinstance(pcas, (MultiPCA, PCA)):
        raise ValueError('pcas must be a PCA or MultiPCA object ')
    return pcas


class PlotlyHelper(object):
    '''Class to help creating plotly plots with shapes, buttons and data'''

    def __init__(self, title, layout):
        self.title = title
        self.layout = layout
        self.data = list()
        self.visibility_map = dict()
        self.updatemenus = list()
        self.shapes = list()
        self.nb_objects = 0
        self.button_group_to_index = dict()
        self.button_group_index = -1

    @staticmethod
    def _get_legend():
        ''' Returns the legend already setup for the plot '''
        return dict(x=0.8, y=1, traceorder='normal',
                    font=dict(family='sans-serif', size=12, color='#000'),
                    bgcolor='#FFFFFF', bordercolor='#FFFFFF', borderwidth=2)

    @staticmethod
    def _get_button_skeleton(direction='down'):
        return dict(type='dropdown', direction=direction, xanchor='left', active=0, buttons=list())

    def _add_button_group(self, name, direction='down'):
        if name not in self.button_group_to_index:
            self.button_group_index += 1
            self.button_group_to_index[name] = self.button_group_index
            if self.button_group_index == 0:
                self.updatemenus = list([self._get_button_skeleton(direction)])
            else:
                self.updatemenus.append(self._get_button_skeleton(direction))

    def _update_visibility(self, name, obj):
        ''' Update the position dictionary for the visibility '''
        if name in self.visibility_map:
            raise ValueError('{} already exists'.format(name))
        self.visibility_map[name] = range(len(self.data), len(self.data) + len(obj))

    def _get_data_positions(self, name):
        ''' Return the positions of the plotly object group named name'''
        return self.visibility_map[name]

    def _place_buttons(self, offset=0.01):
        y = 1
        for group in self.updatemenus:
            group['y'] = y
            gap = 0.04 if group['direction'] == 'right' else len(group['buttons']) * 0.04
            y = y - (gap + offset)

    def get_visibility_map(self, names):
        ''' Return the boolean map for all names'''
        try:
            true_indexes = set([item for name in names for item in self.visibility_map[name]])
        except KeyError as e:
            raise KeyError('Can not find the object {}'.format(e))
        return [i in true_indexes for i in range(self.nb_objects)]

    def add_data(self, obj_group):
        ''' Add data to the plot and update its visibility map '''
        for name, group in obj_group.items():
            if not isinstance(group, list):
                group = [group]
            self._update_visibility(name, group)
            for obj in group:
                self.nb_objects += 1
                self.data.append(obj)

    def add_shape(self, shape):
        ''' Add shape to the figure '''
        self.shapes.append(shape)

    def add_button(self, label, method, args, groupname='classic', direction='down'):
        ''' Add button to the figure.

            Notes:
                Buttons are grouped thanks to the groupname variable
            Args:
                label (str): legend in the button
                method (str): method used by plotly ('restyle', 'relayout', 'update', 'animate')
                args: arguments for the method see plotly documentation
                groupname (str): name of the button group
                direction (str): direction of the group of button ('down', 'left')
        '''
        self._add_button_group(groupname, direction)
        index = self.button_group_to_index[groupname]
        self.updatemenus[index]['buttons'].append(dict(label=label, method=method, args=args))

    def get_fig(self):
        ''' Return the final figure '''
        self._place_buttons()
        self.layout['updatemenus'] = self.updatemenus
        self.layout['shapes'] = self.shapes
        return dict(data=self.data, layout=self.layout)


class PlotlyHelperPlane(PlotlyHelper):
    ''' Helper to create plotly figure on a given plane '''

    def __init__(self, title, plane):
        self.plane = self._sanitize_plane(plane)
        title = self._get_title(title, plane)
        super(PlotlyHelperPlane, self).__init__(title, self._get_layout_skeleton(title, self.plane))

    @staticmethod
    def _get_title(title, plane):
        ''' Return the title with the correct plane '''
        return '{}-{}'.format(title, plane)

    @staticmethod
    def _sanitize_plane(plane):
        ''' sanitizer for the plane input '''
        if not isinstance(plane, six.string_types):
            raise TypeError('plane argument must be a string')
        plane = plane.lower()
        if len(plane) > 2:
            raise ValueError('plane argument must be "3d" or a 2-combination of x, y and z')
        if plane == '3d':
            return 'xyz'
        else:
            values = 'xyz'
            correct = sum([1 if v in plane else 0 for v in values]) == 2
            if not correct or plane[0] == plane[1]:
                raise ValueError('plane argument is not formed of a 2-combination of x, y, and z')
        return plane

    @staticmethod
    def _get_camera(plane):
        ''' Get the default camera for 3d scene or 2d scene'''
        camera = dict(up=dict(x=0, y=0, z=0), center=dict(x=0, y=0, z=0), eye=dict(x=0, y=0, z=0))
        if plane == 'xyz':
            camera['up'] = dict(x=0, y=0, z=1)
            camera['eye'] = dict(x=-1.7428, y=1.0707, z=0.7100, )
        else:
            unit = {v: np.eye(3)[i] for i, v in enumerate('xyz')}
            sign_cross = np.sum(np.sign(np.cross(unit[plane[0]], unit[plane[1]])))
            camera['eye'][list(set('xyz') - set(plane))[0]] = sign_cross * 2
            camera['up'][plane[1]] = 1
        return camera

    @staticmethod
    def _get_scene(plane):
        dragmode = 'zoom' if plane != 'xyz' else 'turntable'
        scene = dict(xaxis=None, yaxis=None, zaxis=None, aspectmode='data',
                     dragmode=dragmode, camera=PlotlyHelperPlane._get_camera(plane))
        for axis in 'xyz':
            axis_name = axis + 'axis'
            scene[axis_name] = dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(238, 238,238)',
                visible=True,
            )
        return scene

    @staticmethod
    def _get_layout_skeleton(title, plane):
        ''' Returns a layout skeleton that is camera compliant '''
        layout = dict(autosize=True, title=title,
                      scene=PlotlyHelperPlane._get_scene(plane),
                      legend=PlotlyHelperPlane._get_legend())
        return layout

    def add_plane_buttons(self):
        if self.plane == 'xyz':
            self.add_button('3D view', 'relayout', ['scene', PlotlyHelperPlane._get_scene('xyz')],
                            'view', 'right')
            self.add_button('XY view', 'relayout', ['scene', PlotlyHelperPlane._get_scene('xy')],
                            'view')
            self.add_button('XZ view', 'relayout', ['scene', PlotlyHelperPlane._get_scene('xz')],
                            'view')
            self.add_button('YZ view', 'relayout', ['scene', PlotlyHelperPlane._get_scene('yz')],
                            'view')


class PCAResults(object):
    ''' Container for PCA results

        Notes :
            This class exists because of a misleading matrix format of the eigv from numpy,
            the lack of a common index representing a given principal component for extent, eigs
            and eigv, the lost of the points' barycenter during the process.
            Using this class allow anyone to safely play with the PCAs.
            The pca importance increase with index : 3rd component index = 0, 2sd = 1, 1st = 2.
    '''

    def __init__(self, extent, eigs, eigv, point_barycenter):
        ''' Init with direct inputs from the principal_direction_extent and pca method'''
        self.extent, self.eigs, self.eigv = self._reorganize_results(extent, eigs, eigv)
        self.center = point_barycenter

    @staticmethod
    def _reorganize_results(extent, eigs, eigv):
        '''reorganize results to create correspondence between extent, eigs, eigv indexes '''
        vectors = sorted([(eig, extent[i], eigv[:, i]) for i, eig in enumerate(eigs)])
        neigs = [v[0] for v in vectors]
        nextent = [v[1] for v in vectors]
        neigv = [v[2] for v in vectors]
        return nextent, neigs, neigv

    def pca_result_iter(self):
        ''' iter to yield extent eigs, eigv in the correct order'''
        for i in range(len(self.eigs)):
            yield self.extent[i], self.eigs[i], self.eigv[i]


class MultiPCA(object):
    ''' A PCA container that ensure a common neuron to run PCAs on '''

    def __init__(self, neuron, pcas):
        self.neuron = neuron
        self.pcas = self.create_pcas(neuron, pcas)
        self.neurite_to_pca_map = self._get_neurite_to_pca_mapping(self.pcas)

    @staticmethod
    def create_pcas(neuron, pcas):
        ''' Class to check and transform input pcas into a list of pca
            Args :
                neurom (morph) : a morphology on which the pca will be processed
                pcas_list (list(dict) or dict or PCA or list(PCA)) : the pca that you want to
                run on the neuron's neurites.

        '''
        if isinstance(pcas, list) and all(isinstance(pca, PCA) for pca in pcas):
            for pca in pcas:
                if not pca.neuron == neuron:
                    raise Exception('You must use the same neuron for all pcas')
            return pcas
        elif isinstance(pcas, list) and all(isinstance(pca, dict) for pca in pcas):
            new_pcas = list()
            name_set = set()
            for pca in pcas:
                if 'name' not in pca or 'group' not in pca:
                    raise Exception('pcas dict must contain: "name", "group" and "axis"')
                if pca['name'] not in name_set:
                    name_set.add(pca['name'])
                    new_pcas.append(PCA(neuron, pca['name'], pca['group'], pca.get('axis', None)))
                else:
                    raise Exception('Two PCAs with the same name or the same group')
            return new_pcas
        elif isinstance(pcas, dict):
            if 'name' not in pcas or 'group' not in pcas or 'axis' not in pcas:
                raise Exception('pcas dict must contain: "name", "group" and "axis"')
            return [PCA(neuron, pcas['name'], pcas['group'], pcas['axis'])]
        elif isinstance(pcas, PCA):
            return [pcas]
        else:
            raise Exception('pcas must be a list of PCA or a list of dict or a PCA')

    @staticmethod
    def _get_neurite_to_pca_mapping(pcas):
        ''' Map the neurite number of the neuron with the corresponding pca '''
        d = dict()
        for i, pca in enumerate(pcas):
            for item in pca.group:
                if item not in d:
                    d[item] = i
                else:
                    raise Exception('Two pcas refer to the same neurite')
        return d

    def get_neurite_to_pca(self, neurite):
        ''' return the pca that corresponds to the neurite '''
        if neurite in self.neurite_to_pca_map:
            return self.pcas[self.neurite_to_pca_map[neurite]]
        return None

    def run_pcas(self):
        ''' Run all PCAs '''
        for pca in self.pcas:
            pca.run_pca()

    def __getitem__(self, item):
        return self.pcas[item]

    def __iter__(self):
        for pca in self.pcas:
            yield pca


class PCA(object):
    ''' Object to store inputs and outputs from PCAs and compute pcas '''

    def __init__(self, neuron, name, group_of_neurites, axis=None):
        self.neuron = neuron
        self.name = name
        self.group = self._sanitize_group(group_of_neurites)
        self.axis = self._sanitize_axis(axis)
        self.is_run = False
        self.res = None

    @staticmethod
    def _sanitize_axis(axis):
        ''' Check the group value. Must be an int or a list of ints '''
        if isinstance(axis, np.ndarray):
            return axis
        elif isinstance(axis, list) and all([isinstance(item, (float, int)) for item in axis]):
            return np.array(axis)
        elif not axis:
            return None
        else:
            raise ValueError('axis must be a list or numpy array of numbers or None')

    def _get_neurite_groups(self, neuron_type):
        ''' Return the neuron's neurites index corresponding to the neuron_type
            Args:
                neuron_type (NeuriteType) : the neuron type needed
            Returns:
                group (list of int) : all neurites coresponding to the neuron_type
        '''
        group = list()
        for i, neurite in enumerate(self.neuron.neurites):
            if neurite.type == neuron_type or neuron_type == NeuriteType.all:
                group.append(i)
        return group

    def _sanitize_group(self, group):
        ''' Check the group value. Must be an int or a list of ints '''
        if isinstance(group, NeuriteType):
            return self._get_neurite_groups(group)
        elif isinstance(group, int):
            return [group]
        elif isinstance(group, list) and all([isinstance(item, int) for item in group]):
            return group
        else:
            raise ValueError('group must be a list of integer or an int')

    def clear_results(self):
        self.is_run = False
        self.res = None

    def run_pca(self):
        ''' Compute the pca for a set of points

        Args:
            neuron (morph) : neuron used to perform the pca

        Returns:
           pca_axes : the plotly object to represent the pca
           pca_res : A PCAResult object that contains usable and ordonated eigs, eigv and extent
        '''
        if not self.is_run:
            self.is_run = True
            points = np.empty((0, 3), dtype=float)
            for neurite in self.group:
                neurite_points = self.neuron.neurites[neurite].points[:, COLS.XYZ]
                points = np.concatenate((points, neurite_points), axis=0)
            pca_extent = principal_direction_extent(points)
            mean = np.mean(points, axis=0)
            eigs, eigv = pca(points - mean)
            self.res = PCAResults(pca_extent, eigs, eigv, mean)


def _make_trace2d(neuron, plane, prefix='', opacity=1., force_color=None, visible=True, style=None):
    '''Create the trace to be plotted'''
    names = defaultdict(int)
    lines = list()
    for neurite in iter_neurites(neuron):
        names[neurite.type] += 1

        if style and neurite in style and 'color' in style[neurite]:
            neurite_color = style[neurite]
        else:
            neurite_color = TREE_COLOR.get(neurite.root_node.type, 'black')

        name = str(neurite.type).replace('NeuriteType.', '').replace('_', ' ')
        name = '{} {} {}'.format(prefix, name, names[neurite.type])

        for section in iter_sections(neurite):
            segs = [(s[0][COLS.XYZ], s[1][COLS.XYZ]) for s in iter_segments(section)]

            if style and section in style and 'color' in style[section]:
                colors = style[section]['color']
            else:
                colors = neurite_color

            coords = dict()
            for i, coord in enumerate('xyz'):
                coords[coord] = list(chain.from_iterable((p1[i], p2[i], None) for p1, p2 in segs))

            coords = dict(x=coords[plane[0]], y=coords[plane[1]])
            lines.append(go.Scattergl(name=name, visible=visible, opacity=opacity,
                                      line=dict(color=colors, width=2),
                                      mode='lines',
                                      **coords))
    return lines


def _make_trace(neuron, plane, prefix='', opacity=1., force_color=None, visible=True, style=None):
    '''Create the trace to be plotted'''
    names = defaultdict(int)
    lines = list()
    for neurite in iter_neurites(neuron):
        names[neurite.type] += 1

        coords = dict(x=list(), y=list(), z=list())
        colors = list()

        default_color = TREE_COLOR.get(neurite.root_node.type, 'black')

        for section in iter_sections(neurite):
            segs = [(s[0][COLS.XYZ], s[1][COLS.XYZ]) for s in iter_segments(section)]

            if style and section in style and 'color' in style[section]:
                start_index = style[section].get('index_offset', 0)
                start_index = min(start_index, len(segs))
                colors += list(repeat(default_color, 3*start_index))
                colors += list(repeat(style[section]['color'], 3*(len(segs) - start_index)))
            else:
                colors += list(repeat(default_color, 3*len(segs)))

            for i, coord in enumerate('xyz'):
                if coord in plane:
                    coords[coord] += list(chain.from_iterable((p1[i], p2[i], None)
                                                              for p1, p2 in segs))
                else:
                    coords[coord] += list(chain.from_iterable((0, 0, None) for _ in segs))

        name = str(neurite.type).replace('NeuriteType.', '').replace('_', ' ')
        name = '{} {} {}'.format(prefix, name, names[neurite.type])
        lines.append(go.Scatter3d(name=name, visible=visible, opacity=opacity,
                                  line=dict(color=colors, width=2),
                                  mode='lines',
                                  **coords))
    return lines


def _make_soma(neuron):
    ''' Create a 3d surface representing the soma '''
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 100)
    z = np.outer(np.ones(100), np.cos(phi)) + neuron.soma.center[2]
    r = neuron.soma.radius
    return go.Surface(
        name=SOMA_NAME,
        x=(np.outer(np.cos(theta), np.sin(phi)) + neuron.soma.center[0]) * r,
        y=(np.outer(np.sin(theta), np.sin(phi)) + neuron.soma.center[1]) * r,
        z=z * r,
        cauto=False, surfacecolor=['black'] * len(z), showscale=False,
    )


def _make_pca(pca, center, plane, visible=True):
    ''' Create plotly axes for the pca axes. PCA must be run before calling this fuction

            Args:
                pca (PCA): pca object you want to display
                plane (str): plane for the vector representation similar to the draw function
                visible (boolean): set the visibility of the axes

            Returns:
                 pca_axes (list of Scatter3d) the pca axes with correct colors and size
            '''
    if not pca.is_run:
        msg = 'Cannot create plotly axes for {} if the pca is not run before'.format(pca.name)
        raise Exception(msg)
    pca_axes = list()
    for i, (extent, _, eigv) in enumerate(pca.res.pca_result_iter()):
        color, pca_name = PCA_AXIS_INFOS[i]
        coords = dict(x=None, y=None, z=None)
        for i, coord in enumerate('xyz'):
            if coord in plane:
                coords[coord] = [center[i], eigv[i] * extent + center[i]]
            else:
                coords[coord] = [0, 0]
        pca_axes.append(go.Scatter3d(name=pca_name, visible=visible,
                                     marker=dict(size=2, color=color, symbol='diamond'),
                                     line=dict(color=color, width=5), **coords))
    return pca_axes


class PlotBuilder:
    def __init__(self, neuron, plane, title, inline):
        self.title = title
        self.neuron = neuron
        self.inline = inline
        self.plane = plane

        self.properties = defaultdict(dict)
        self.helper = PlotlyHelperPlane(self.title, self.plane)

    def color_section(self, section, color, recursive=False, index_offset=0):
        self.properties[section]['color'] = color
        if index_offset:
            self.properties[section]['index_offset'] = index_offset
        if recursive:
            for child in section.children:
                self.color_section(child, color, recursive=True, index_offset=0)

    def plot(self, *args, **kwargs):
        self.helper.add_data({NEURON_NAME: _make_trace(
            self.neuron, self.helper.plane, style=self.properties)})
        self.helper.add_data({SOMA_NAME: _make_soma(self.neuron)})
        self.helper.add_plane_buttons()
        fig = self.helper.get_fig()
        plot_fun = iplot if self.inline else plot
        if self.inline:
            init_notebook_mode(connected=True)  # pragma: no cover
        plot_fun(fig, filename=self.title + '.html', *args, **kwargs)
        return fig


def _plot_neuron(neuron, plane, title, inline, **kwargs):
    ''' Draw a neuron using plotly '''
    builder = PlotBuilder(neuron, plane, title, inline, **kwargs)
    return builder.plot()
