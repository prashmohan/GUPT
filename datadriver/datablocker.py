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
import time
from common import *

logger = logging.getLogger(__name__)
    
class DataBlockerFactory(object):
    @staticmethod
    def get_blocker(blocker_name):
        for klass in GuptDataBlocker.__subclasses__():
            if klass.__name__ == blocker_name:
                return klass
        return None

    @staticmethod
    def get_blocker_names():
        return [klass.__name__ for klass in GuptDataBlocker.__subclasses__()]


class GuptDataBlocker(object):
    """
    Base class that defines how to sample data records into multiple
    blocks
    """
    def __init__(self, args):
        self.args = args
        
    def get_blocks(self, records):
        """
        Convert the given set of records into multiple smaller blocks
        of data records
        """
        raise Exception("This function should be over ridden")

    def get_new_epsilon(self, epsilon):
        """
        Return the new epsilon for the compuation based on the
        blocking technique used
        """
        return epsilon

    
class NaiveDataBlocker(GuptDataBlocker):
    def get_blocks(self, records):
        """
        Convert the given set of records into multiple smaller blocks
        of data records
        """
        logger.debug("Using " + self.__class__.__name__ + " for data blocking")
        num_records = len(records)
        num_blocks = int(num_records ** 0.4)
        block_size = int(math.ceil(num_records / num_blocks))
        logger.info("Num Records: %d, Block size: %d, Num blocks: %d" %
                    (num_records, block_size, num_blocks))
        return [records[indices : indices + block_size] for indices in range(0, num_records, block_size)]


class ResamplingBlocker(object):
    @staticmethod
    def get_blocks_gamma(records, num_blocks, block_size, gamma):
        """
        Subsample from the set of records into num_blocks distinct
        blocks. Distribute each record among gamma of the subsampled
        blocks (each of which is of size block_size)
        """
        logger.info("Num Records: %d, Block size: %d, Num blocks: %d, gamma: %d" %
                    (len(records), block_size, num_blocks, gamma))
        # Size of each block does not change
        blocks = []
        for x in range(num_blocks):
            blocks.append([])
        nonfull_blocks = range(num_blocks)
        
        for record in records:
            for block_no in random.sample(nonfull_blocks, gamma):
                blocks[block_no].append(record)
                if len(blocks[block_no]) >= block_size:
                    # Remove block from contenders list if block is
                    # already full
                    nonfull_blocks.remove(block_no)
        return blocks
    

class ResamplingDataBlockerConstantSize(GuptDataBlocker):
    @profile_func
    def get_blocks(self, records):
        logger.debug("Using " + self.__class__.__name__ + " for data blocking")
        num_records = len(records)
        num_blocks = self.args[0] * int(math.ceil(num_records ** 0.4))
        # Each record is put into num_blocks blocks. But each element
        # is repeated in gamma blocks. Thus the total number of blocks
        # becomes gamma * original number of block
        block_size = int(math.ceil(num_records ** 0.6))
        return ResamplingBlocker.get_blocks_gamma(records, num_blocks, block_size, self.args[0])

    def get_new_epsilon(self, epsilon):
        """
        Return the new epsilon for the compuation based on the
        blocking technique used
        """
        return float(epsilon) / self.args[0]
    

class ResamplingDataBlockerConstantBlocks(GuptDataBlocker):
    @profile_func
    def get_blocks(self, records):
        """
        Convert the given set of records into multiple smaller blocks
        of data records
        """
        logger.debug("Using " + self.__class__.__name__ + " for data blocking")
        num_records = len(records)
        num_blocks = int(math.ceil(num_records ** 0.4))
        # Each record is put into gamma blocks. Thus the total size of
        # each blocks becomes gamma * original size of block
        block_size = self.args[0] * int(math.ceil(num_records ** 0.6))
        return ResamplingBlocker.get_blocks_gamma(records, num_blocks, block_size, self.args[0])

    def get_new_epsilon(self, epsilon):
        """
        Return the new epsilon for the compuation based on the
        blocking technique used
        """
        return float(epsilon) / self.args[0]
