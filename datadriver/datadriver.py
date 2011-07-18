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

logger = logging.getLogger(__name__)

class GuptDataDriver(object):
    """
    This class should be subclassed by the Data Provider. The data
    provider will need to provide the logic for parsing the records
    from the data sources. The data sources itself could be from
    files, databases, etc. A sample data driver in the form of
    CSVDriver is provided along with Gupt. Finally, the data provider
    will also need to provide bounds on the inputs.
    """    
    def __init__(self, filter=None, transformer=None):
        """
        The filter is an optional argument that restricts the number
        of tuples under study.
        """
        self.filter = filter
        self.transformer = transformer

    def set_data_source(self, *fargs):
        """
        This sets and initializes the data sources. Typically this
        would mean setting the file name or the database connection
        """
        raise Exception("This function should be over ridden")

    def set_input_bounds(self, bounds):
        self.min_bounds, self.max_bounds = zip(*bounds)

    def set_sensitiveness(self, sensitiveness):
        self.sensitiveness = sensitiveness

    def create_record(self):
        """
        Generate the next record
        """
        raise Exception("This function should be over ridden")

    def get_next_record(self):
        record = self.create_record()
        if not record:
            return None
        if self.filter and not self.filter(record):
            return None
        if self.transformer:
            return self.transformer(record)
        return record

    def get_records(self):
        """
        Retrieve all records
        """
        rec = self.get_next_record()
        records = []
        while rec:
            records.append(rec)
            rec = self.get_next_record()
        return records
    
    
if __name__ == '__main__':
    print >> sys.stderr, "This is a library and should not be executed standalone"
    sys.exit(1)
