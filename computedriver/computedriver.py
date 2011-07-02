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

log = logging.getLogger(__name__)

class GuptComputeDriver(object):
    """
    This class should be subclassed by the Computation Provider. All
    of the computation should be encapsulated in a class that
    subclasses GuptComputeDriver. The `initialize' function of the
    computation class will be invoked upon load. Subsequently, the
    `execute' function will be executed for each record and the
    `finalize' function will be executed before the termination of the
    program. The output from any and all of these functions will be
    interpreted as part of the output of the program.
    """
    def initialize(self):
        """
        Optionally implemented to initalize the computation
        """
        pass

    def execute(self, record):
        """
        Must be overridden to provide execution logic for each record
        """
        raise logging.exception("This function should be over ridden")

    def finalize(self):
        """
        Optionally implemented to handle the termination of the program
        """
        pass

    def get_output_bounds(self, first_quartile, third_quartile):
        """
        Retrieve the bounds on the output for the computation
        """
        raise logging.exception("This function should be over ridden")

    def get_input_bounds(self):
        """
        Retrieve the bounds on the input for the computation
        """
        raise logging.exception("This function should be over ridden")


