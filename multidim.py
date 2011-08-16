#!/usr/bin/env python
"""
Author:  Prashanth Mohan <prashmohan@gmail.com>
         http://www.cs.berkeley.edu/~prmohan
        
Copyright (c) 2011, University of California at Berkeley
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of University of California, Berkeley nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE UNIVERSITY
OF CALIFORNIA AT BERKELEY BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
"""

from common import *

class MultiDimensional(object):
    @staticmethod
    def avg(vectors):
        """
        Perform multidimensional averaging of outputs
        """
        temp_vec = vectors[0]
        for vector in vectors[1:]:
            temp_vec = MultiDimensional.add(temp_vec, vector)

        return MultiDimensional.div_scalar(temp_vec, len(vectors))

    @staticmethod
    def add(vec_a, vec_b):
        if not isiterable(vec_a) or not isiterable(vec_b):
            return vec_a + vec_b

        return [MultiDimensional.add(vec_a[index], vec_b[index]) for index in range(len(vec_a))]

    @staticmethod
    def sub(vec_a, vec_b):
        if not isiterable(vec_a) or not isiterable(vec_b):
            return vec_a - vec_b

        return [MultiDimensional.sub(vec_a[index], vec_b[index]) for index in range(len(vec_a))]

    @staticmethod
    def abs(vec_a):
        if not isiterable(vec_a):
            return abs(vec_a)

        return [MultiDimensional.abs(vec_a[index]) for index in range(len(vec_a))]

    @staticmethod
    def mul(vec_a, vec_b):
        if not isiterable(vec_a) or not isiterable(vec_b):
            return vec_a * vec_b

        return [MultiDimensional.mul(vec_a[index], vec_b[index]) for index in range(len(vec_a))]

    @staticmethod
    def div(vec_a, vec_b):
        if not isiterable(vec_a) or not isiterable(vec_b):
            return vec_a / vec_b

        return [MultiDimensional.div(vec_a[index], vec_b[index]) for index in range(len(vec_a))]

    @staticmethod
    def add_scalar(vec_a, scalar):
        if not isiterable(vec_a):
            return vec_a + scalar

        return [MultiDimensional.add_scalar(vec_a[index], scalar) for index in range(len(vec_a))]

    @staticmethod
    def sub_scalar(vec_a, scalar):
        if not isiterable(vec_a):
            return vec_a - scalar

        return [MultiDimensional.sub_scalar(vec_a[index], scalar) for index in range(len(vec_a))]

    @staticmethod
    def mul_scalar(vec_a, scalar):
        if not isiterable(vec_a):
            return vec_a * scalar

        return [MultiDimensional.mul_scalar(vec_a[index], scalar) for index in range(len(vec_a))]

    @staticmethod
    def div_scalar(vec_a, scalar):
        if not isiterable(vec_a):
            return vec_a / scalar

        return [MultiDimensional.div_scalar(vec_a[index], scalar) for index in range(len(vec_a))]

    @staticmethod
    def zip(*vectors):
        """
        Perform the functionality of the zip builtin when there is
        more than 2 dimensions
        """
        for d in vectors:
            if not isiterable(d):
                return vectors
        
        return [MultiDimensional.zip(*d) for d in zip(*vectors)]
        
