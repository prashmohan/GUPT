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

import sys
import os
import logging
import math
import random
import numpy as np

logger = logging.getLogger(__name__)

def histogram(records_transpose, sensitive, epsilon):
    logger.debug("Estimating the distribution of values")
    hist = []
    for index, column in enumerate(records_transpose):
        if not sensitive[index]:
            hist.append(None)
        else:
            hist.append(_get_dp_hist(column, epsilon))
    return hist

def _get_dp_hist(records, epsilon):
    records = np.array(records)
    TOTAL_BUCKETS = 10
    counts = np.histogram(records, bins=TOTAL_BUCKETS)

    hist = np.array(map(float, counts[0]))
    for x in range(len(hist)):
        hist[x] += gen_noise(1.0 / epsilon)
        hist[x] /= len(records)
    return hist    
    
def estimate_percentile(percentile, records, epsilon, min_val, max_val):
    """
    Perform a differentially private percentile estimation based on
    Section 3.2 of "Discovering frequent patterns in sensitive data"
    by R. Bhaskar et. al
    """
    vals = [min_val, max_val]
    vals.extend(records)
    for index, val in enumerate(vals):
        if val > max_val:
            vals[index] = max_val
        elif val < min_val:
            vals[index] = min_val
    vals.sort()

    k = len(vals) - 2
    inv_half_epsilon = 2.0 / epsilon

    q = [(-1.0 * abs(index - percentile * k)) + \
             inv_half_epsilon * np.log(vals[index + 1] - vals[index]) + \
             gen_noise(inv_half_epsilon)
         for index in range(len(vals) - 1)]
        
    picked = max(xrange(len(q)), key=q.__getitem__) # Pick the index of the largest element in the sequence
    return random.uniform(vals[picked], vals[picked + 1])

def _sign(number):
    """
    Returns the sign of the number.
    -1 if number < 0, 0 if number == 0 and +1 if number > 0
    """
    return cmp(number, 0)

def gen_noise(scale):
    """
    Generate a Laplacian noise to satisfy differential privacy
    """
    uniform = random.random() - 0.5
    return scale * _sign(uniform) * math.log(1 - 2.0 * abs(uniform))
