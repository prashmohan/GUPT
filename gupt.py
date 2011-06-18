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
import datadriver.datadriver as datadriver

# Log verbosely
root_logger = logging.getLogger('')
root_logger.setLevel(logging.DEBUG)

# Logger console output
# console = logging.StreamHandler(sys.stderr)
# console_format = '%(message)s'
# console.setFormatter(logging.Formatter(console_format))
# console.setLevel(logging.INFO)
# root_logger.addHandler(console)

# Traceback handlers
traceback_log = logging.getLogger('traceback')
traceback_log.propogate = False
traceback_log.setLevel(logging.ERROR)

if __name__ == '__main__':
    # Logger file output
    file_handler = logging.handlers.RotatingFileHandler(sys.argv[0] + '.log', )
    file_format = '%(asctime)s %(levelname)6s %(name)s %(message)s'
    file_handler.setFormatter(logging.Formatter(file_format))
    file_handler.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    traceback_log.addHandler(file_handler)


def handle_errors(exc_type, exc_value, traceback):
    logging.getLogger(__name__).error(exc_value)
    logging.getLogger('traceback').error(
        exc_value,
        exc_info=(exc_type, exc_value, traceback),
        )
sys.excepthook = handle_errors

log = logging.getLogger(__name__)


class GuptException(Exception):
    pass


class GuptRunTime(object):
    """
    This class defines the runtime for GUPT. It requires a DataDriver
    and a ComputeDriver in order to operate. It then feeds in data
    from the DataDriver to the computation and finally estimates the
    noise required to guarantee differential privacy.
    """
    def __init__(self, compute_driver, data_driver):
        self.compute_driver = compute_driver
        self.data_driver = data_driver
        if not isinstance(compute_driver, GuptComputeDriver):
            raise GuptException("Argument compute_driver is not subclassed from GuptComputeDriver")
        if not isinstance(data_driver, datadriver.GuptDataDriver):
            raise GuptException("Argument data_driver is not subclassed from GuptDataDriver")

    def start(self):
        pass
    

class GuptComputeDriver(object):
    """
    This class should be subclassed by the Computation Provider. All
    of the computation should be encapsulated in a class that
    subclasses GuptComputeDriver. The `initialize' function of the
    computation class will be invoked upon load. Subsequently, the
    `exec' function will be executed for each record and the
    `finalize' function will be executed before the termination of the
    program. The output from any and all of these functions will be
    interpreted as part of the output of the program.
    """
    def initialize(self):
        """
        Optionally implemented to initalize the computation
        """
        pass

    def exec(self):
        """
        Must be overridden to provide execution logic for each record
        """
        raise GuptException("This function should be over ridden")

    def finalize(self):
        """
        Optionally implemented to handle the termination of the program
        """
        pass


if __name__ == '__main__':
    print >> sys.stderr, "This is a library and should not be executed standalone"
    sys.exit(1)
