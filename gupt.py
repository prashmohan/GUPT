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
import logging.handlers
import random
import types
import math
import time
import dpalgos
from datadriver.datadriver import GuptDataDriver
from computedriver.computedriver import GuptComputeDriver

# Log verbosely
root_logger = logging.getLogger('')
root_logger.setLevel(logging.DEBUG)

# Logger console output
console = logging.StreamHandler(sys.stderr)
console_format = '%(levelname)6s %(name)s: %(message)s'
console.setFormatter(logging.Formatter(console_format))
console.setLevel(logging.INFO)
root_logger.addHandler(console)

# Traceback handlers
traceback_log = logging.getLogger('traceback')
traceback_log.propogate = False
traceback_log.setLevel(logging.ERROR)

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

class GuptOutput(object):
    def __init__(self):
        self.output = []

    def append(self, record):
        if not record:
            return
        if type(record) == types.ListType:
            self.output.extend(record)
        else:
            self.output.append(record)

    def __len__(self):
        return len(self.output)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return str(self.output)

    def __iter__(self):
        return iter(self.output)
            

class GuptRunTime(object):
    """
    This class defines the runtime for GUPT. It requires a DataDriver
    and a ComputeDriver in order to operate. It then feeds in data
    from the DataDriver to the computation and finally estimates the
    noise required to guarantee differential privacy.
    """
    def __init__(self, compute_driver_class, data_driver, epsilon):
        self.epsilon = float(epsilon)
        if not issubclass(compute_driver_class, GuptComputeDriver):
            raise logging.exception("Argument compute_driver is not subclassed from GuptComputeDriver")
        if not isinstance(data_driver, GuptDataDriver):
            raise logging.exception("Argument data_driver is not subclassed from GuptDataDriver")
        self.compute_driver_class = compute_driver_class
        self.data_driver = data_driver

    def sign(self, number):
        """
        Returns the sign of the number.
        -1 if number < 0, 0 if number == 0 and +1 if number > 0
        """
        return cmp(number, 0)

    def gen_noise(self, scale):
        """
        Generate a Laplacian noise to satisfy differential privacy
        """
        uniform = random.random() - 0.5
        return scale * self.sign(uniform) * math.log(1 - 2.0 * abs(uniform))

    def start(self):
        """
        Start the differentially private data analysis
        """
        logging.debug("Initializing the differentially private data analysis for " +
                     str(self.compute_driver_class) + " on " +
                     str(self.data_driver))
        
        # Retrieve the input records
        start_time = time.time()
        records = self.data_driver.get_records()
        logging.debug("Finished reading all records: " + str(time.time() - start_time))

        # Obtain the output bounds on the data
        start_time = time.time()
        lower_bounds, higher_bounds = self.get_data_bounds(records, self.epsilon)
        logging.debug("Finished generating the bounds: " + str(time.time() - start_time))
        logging.info("Output bounds are %s and %s" % (str(lower_bounds), str(higher_bounds)))
        
        # Execute the various intances of the computation
        logging.info("Initializing execution of data analysis")
        start_time = time.time()
        outputs = self.execute(records)
        logging.debug("Finished executing the computation: " + str(time.time() - start_time))
        
        # Ensure that the output dimension was the same for all
        # instances of the computation
        lengths = set([len(output) for output in outputs])
        lens_peek = lengths.pop()
        lengths.add(lens_peek)
        if len(lengths) > 1 or lens_peek != len(higher_bounds) or \
                lens_peek != len(lower_bounds):
            raise logging.exception("Output dimension is varying for each instance of the computation")

        # Aggregate and privatize the final output from various instances
        final_output = self.privatize(self.epsilon / (3 * len(outputs[0])), \
                                          lower_bounds, higher_bounds, outputs)
        return final_output
        
    def get_data_bounds(self, records, epsilon):
        """
        Generate the output bounds for the given data set for a pre
        defined computation
        """
        compute_driver = self.compute_driver_class()
        min_vals, max_vals = self.data_driver.min_bounds, self.data_driver.max_bounds
        sensitive = self.data_driver.sensitiveness

        # Find the first and third quartile of the distribution in a
        # differentially private manner
        records_transpose = zip(*records)
        logging.debug("Estimating quartiles")
        first_quartile = []
        third_quartile = []
        for index in range(len(records_transpose)):
            if not sensitive[index]:
                first_quartile.append(0)
                third_quartile.append(0)
            else:
                fq = dpalgos.estimate_percentile(0.25,
                                                 records_transpose[index],
                                                 epsilon,
                                                 min_vals[index],
                                                 max_vals[index])
                tq = dpalgos.estimate_percentile(0.75,
                                                 records_transpose[index],
                                                 epsilon,
                                                 min_vals[index],
                                                 max_vals[index])
                first_quartile.append(fq)
                third_quartile.append(tq)

        logging.debug("Finished quartile estimation")
        logging.debug("Output bound estimation in progress")
        # Use the ComputeDriver's bound generator to generate the
        # output bounds

        return compute_driver.get_output_bounds(first_quartile,
                                                third_quartile)
    
    def execute(self, records):
        """
        Execute the computation provider in a differentially private
        manner for the given set of records.
        """
        num_records = len(records)
        # TODO: Check if we can use random.sample instead of
        # shuffle. Heavy performance benefits.
        random.shuffle(records)
        num_blocks = int(num_records ** 0.4)
        block_size = int(math.ceil(num_records / num_blocks))
        logging.info("Num Records: %d, Block size: %d, Num blocks: %d" %
                     (num_records, block_size, num_blocks))
        outputs = []

        for indices in range(0, num_records, block_size):
            logging.debug("Starting data analytics on block no %d, indices %d - %d" %
                          (indices / block_size, indices, indices + block_size))
            compute_driver = self.compute_driver_class()
            cur_output = GuptOutput()
            cur_output.append(compute_driver.initialize())
            for record in records[indices : indices + block_size]:
                cur_output.append(compute_driver.execute(record))
            cur_output.append(compute_driver.finalize())
            outputs.append(cur_output)
        return outputs

    def privatize(self, epsilon, lower_bounds, higher_bounds, outputs):
        """
        Converts the output of many instances of the computation
        into a differentially private answer
        """
        bound_ranges = []
        for index in range(len(lower_bounds)):
            bound_ranges.append(abs(lower_bounds[index] - higher_bounds[index]))
        
        # Take the average of the outputs of each instance of the
        # computation
        final_output = [0.0] * len(outputs[0])
        for output in outputs:
            for index, val in enumerate(output):
                final_output[index] += val

        # Add a Laplacian noise in order to ensure differential privacy
        for index in range(len(final_output)):
            final_output[index] = final_output[index] / len(outputs)
            logging.info("Final Answer (Unperturbed) Dimension %d = %f" % (index, final_output[index]))
            noise = self.gen_noise(bound_ranges[index] / (epsilon * len(outputs)))
            logging.info("Perturbation = " + str(noise))
            final_output[index] += noise
            logging.info("Final Answer (Perturbed) Dimension %d = %f" % (index, final_output[index]))
        return final_output
            

if __name__ == '__main__':
    print >> sys.stderr, "This is a library and should not be executed standalone"
    sys.exit(1)
