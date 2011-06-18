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
import datadriver
import csv

log = logging.getLogger(__name__)

class CSVDriver(GuptDataDriver):
    """
    This is a sample data driver converts CSV files into data tuples.
    """
    def __init__(self, filter=None, skip=0):
        self.skip = skip
        super(CSVDriver, self).__init__(filter)
        
    def set_data_source(self, *fargs):
        self.csv_file_path = fargs[0]
        self.csv_file = csv.reader(open(self.csv_file_path, 'r'))

    def get_next_record(self):
        record = self.csv_file.next()
        if self.filter and not self.filter(record):
            return None
        return record
    
    
if __name__ == '__main__':
    print >> sys.stderr, "This is a library and should not be executed standalone"
    sys.exit(1)
