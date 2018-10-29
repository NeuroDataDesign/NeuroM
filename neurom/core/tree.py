# Copyright (c) 2015, Ecole Polytechnique Federale de Lausanne, Blue Brain Project
# All rights reserved.
#
# This file is part of NeuroM <https://github.com/BlueBrain/NeuroM>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#     3. Neither the name of the copyright holder nor the names of
#        its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''Generic tree class and iteration functions'''
import numpy as np
from collections import deque

from neurom._compat import filter

from neurom import morphmath
from neurom.core.dataformat import COLS


class Section(object):
    '''Simple recursive tree class'''

    def __init__(self, morphio_section):
        self.id = morphio_section.id
        self.morphio_section = morphio_section

    @property
    def parent(self):
        return None if self.morphio_section.is_root else Section(self.morphio_section.parent)

    @property
    def children(self):
        return [Section(child) for child in self.morphio_section.children]

    def is_forking_point(self):
        '''Is tree a forking point?'''
        return len(self.children) > 1

    def is_bifurcation_point(self):
        '''Is tree a bifurcation point?'''
        return len(self.children) == 2

    def is_leaf(self):
        '''Is tree a leaf?'''
        return len(self.children) == 0

    def is_root(self):
        '''Is tree the root node?'''
        return self.parent is None

    def ipreorder(self):
        '''Depth-first pre-order iteration of tree nodes'''
        children = deque((self, ))
        while children:
            cur_node = children.pop()
            children.extend(reversed(cur_node.children))
            yield cur_node

    def ipostorder(self):
        '''Depth-first post-order iteration of tree nodes'''
        children = [self, ]
        seen = set()
        while children:
            cur_node = children[-1]
            if cur_node not in seen:
                seen.add(cur_node)
                children.extend(reversed(cur_node.children))
            else:
                children.pop()
                yield cur_node

    def iupstream(self):
        '''Iterate from a tree node to the root nodes'''
        t = self
        while t is not None:
            yield t
            t = t.parent

    def ileaf(self):
        '''Iterator to all leaves of a tree'''
        return filter(Tree.is_leaf, self.ipreorder())

    def iforking_point(self, iter_mode=ipreorder):
        '''Iterator to forking points. Returns a tree object.

        Parameters:
            tree: the tree over which to iterate
            iter_mode: iteration mode. Default: ipreorder.
        '''
        return filter(Tree.is_forking_point, iter_mode(self))

    def ibifurcation_point(self, iter_mode=ipreorder):
        '''Iterator to bifurcation points. Returns a tree object.

        Parameters:
            tree: the tree over which to iterate
            iter_mode: iteration mode. Default: ipreorder.
        '''
        return filter(Tree.is_bifurcation_point, iter_mode(self))

    def __nonzero__(self):
        return bool(self.children)

    __bool__ = __nonzero__

    @property
    def points(self):
        return np.concatenate((self.morphio_section.points,
                               self.morphio_section.diameters[:, np.newaxis] / 2.),
                              axis=1)

    @points.setter
    def points(self, value):
        self.morphio_section.points = np.copy(value[:, COLS.XYZ])
        self.morphio_section.diameters = np.copy(value[:, COLS.R]) * 2

    @property
    def type(self):
        return self.morphio_section.type

    # TODO: Should we have a @type.setter ?

    @property
    #@memoize
    def length(self):
        '''Return the path length of this section.'''
        return morphmath.section_length(self.points)

    @property
    #@memoize
    def area(self):
        '''Return the surface area of this section.

        The area is calculated from the segments, as defined by this
        section's points
        '''
        return sum(morphmath.segment_area(s) for s in iter_segments(self))

    @property
    #@memoize
    def volume(self):
        '''Return the volume of this section.

        The volume is calculated from the segments, as defined by this
        section's points
        '''
        return sum(morphmath.segment_volume(s) for s in iter_segments(self))

    def __str__(self):
        return 'Section(id=%s, type=%s, n_points=%s) <parent: %s, nchildren: %d>' % \
            (self.id, self.type, len(self.points), self.parent, len(self.children))

    __repr__ = __str__


Tree = Section
