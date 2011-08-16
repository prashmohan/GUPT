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
from multiprocessing import Process, Pipe
from itertools import izip

from multidim import MultiDimensional
import dpalgos
from common import *
from datadriver.datadriver import GuptDataDriver
from datadriver.datablocker import DataBlockerFactory
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

logger = logging.getLogger(__name__)

# Alternate implementation to multiprocessing.Pool.map, since it has
# many issues with pickling of functions and methods. Below code spawn
# and parmap taken from
# http://stackoverflow.com/questions/3288595/multiprocessing-using-pool-map-on-a-function-defined-in-a-class/5792404#5792404
def spawn(f):
    def func(pipe, *x):
        pipe.send(f(*x))
        pipe.close()
    return func

def parmap(f, X):
    pipe = [Pipe() for x in X]
    proc = [Process(target=spawn(f), args=(c, x)) for x, (p, c) in izip(X, pipe)]
    [p.start() for p in proc]
    [p.join() for p in proc]
    return [p.recv() for (p, c) in pipe]

# End of parallel map implementation

class GuptOutput(object):
    def __init__(self):
        self.output = []

    def append(self, record):
        if record == None:
            return
        else:
            self.output.append(record)

    def extend(self, records):
        if not isiterable(records):
            self.append(records)
        else:
            self.output.extend(records)

    def __len__(self):
        return len(self.output)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return str(self.output)

    def __iter__(self):
        return iter(self.output)

    def __getitem__(self, index):
        return self.output[index]

    def __setitem__(self, index, value):
        self.output[index] = value


class GuptRunTime(object):
    """
    This class defines the runtime for GUPT. It requires a DataDriver
    and a ComputeDriver in order to operate. It then feeds in data
    from the DataDriver to the computation and finally estimates the
    noise required to guarantee differential privacy.
    """
    def __init__(self, compute_driver_class, data_driver, epsilon,
                 blocker_name='NaiveDataBlocker', blocker_args=None):
        
        if not issubclass(compute_driver_class, GuptComputeDriver):
            raise log.exception("Argument compute_driver is not subclassed from GuptComputeDriver")
        if not isinstance(data_driver, GuptDataDriver):
            raise logger.exception("Argument data_driver is not subclassed from GuptDataDriver")
        
        self.compute_driver_class = compute_driver_class
        self.data_driver = data_driver
        
        if not blocker_args or isiterable(blocker_args):
            self.blocker_args = blocker_args
        else:
            self.blocker_args = (blocker_args, )
        self.blocker = DataBlockerFactory.get_blocker(blocker_name)(self.blocker_args)
        
        self.sensitivity_factor = self.blocker.get_sensitivity_factor()
        logger.debug("The sensitivity of the output has changed by a factor of %f because of blocking" %
                     (self.sensitivity_factor))
        self.epsilon = float(epsilon)

        logger.info("Initializing Gupt Runtime environment for analysis of the " +
                    str(data_driver) + " data set " +
                    "using the " + compute_driver_class.__name__ + " computation " +
                    "with an epsilon value of " + str(epsilon))                    
        
    @staticmethod
    def get_data_blockers():
        return DataBlockerFactory.get_blocker_names()

    def _windsorized(self, epsilon, lower_bounds, higher_bounds, output):
        """
        Privatize each dimension of the output in a winsorized manner
        """
        if isiterable(output[0]):
            noise = []
            estimate = []
            for index in range(len(output)):
                e, n = self._windsorized(epsilon / len(output), lower_bounds[index], higher_bounds[index], output[index])
                estimate.append(e)
                noise.append(n)
            return estimate, noise

        dimension = list(output)
        rad = len(output) ** (1.0 / 3 + 0.1)
        
        lps = dpalgos.estimate_percentile(0.25, dimension,
                                          epsilon / 4,
                                          lower_bounds,
                                          higher_bounds)
        hps = dpalgos.estimate_percentile(0.75, dimension,
                                          epsilon / 4,
                                          lower_bounds,
                                          higher_bounds)
        crude_mu = float(lps + hps) / 2
        crude_iqr = abs(hps - lps)
        u = crude_mu + 4 * rad * crude_iqr
        l = crude_mu - 4 * rad * crude_iqr
        # Compute windsorized mean for range
        self._sanitize_multidim(dimension, [l] * len(dimension), [u] * len(dimension))
                
        mean_estimate = float(sum(dimension)) / len(dimension)
        noise = dpalgos.gen_noise(self.sensitivity_factor * float(abs(u - l)) / (2 * epsilon * len(dimension)))
        return mean_estimate, noise

    @profile_func
    def _privatize_windsorized(self, epsilon, lower_bounds, higher_bounds, outputs):
        outputs_transpose = MultiDimensional.zip(*outputs)

        final_output = []
        # Add a Laplacian noise in order to ensure differential privacy
        for index, dimension in enumerate(outputs_transpose):
            estimate, noise = self._windsorized(epsilon, lower_bounds[index], higher_bounds[index], dimension)
            logger.info("Final Answer (Unperturbed) Dimension " + str(index) + " = " + str(estimate))
            logger.info("Perturbation = " + str(noise))
            final_output.append(MultiDimensional.add(estimate, noise))
            logger.info("Final Answer (Perturbed) Dimension " + str(index) + " = " + str(final_output[-1]))
            
        return final_output

    @profile_func
    def _start_nonprivate_analysis(self):
        """
        Start a non private analysis on the data set
        """
        logger.debug("Initializing the non-private data analysis for " +
                     str(self.compute_driver_class) + " on " +
                     str(self.data_driver))
        
        # Retrieve the input records
        start_time = time.time()
        records = self.data_driver.get_records()
        logger.debug("Finished reading all records: " + str(time.time() - start_time))

        # Execute the various intances of the computation
        logger.info("Initializing execution of data analysis")
        start_time = time.time()
        outputs = self._apply_compute_driver(records)
        logger.debug("Finished executing the computation: " + str(time.time() - start_time))

        return outputs

    @profile_func
    def _start_diff_analysis(self, ret_bounds, sanitize, privatize):
        """
        Start the differentially private data analysis
        """
        logger.debug("Initializing the differentially private data analysis for " +
                     str(self.compute_driver_class) + " on " +
                     str(self.data_driver))
        
        # Retrieve the input records
        start_time = time.time()
        records = self.data_driver.get_records()
        logger.debug("Finished reading all records: " + str(time.time() - start_time))

        # Obtain the output bounds on the data
        start_time = time.time()
        lower_bounds, higher_bounds = ret_bounds(records, self.epsilon)
        logger.debug("Finished generating the bounds: " + str(time.time() - start_time))
        logger.info("Output bounds are %s and %s" % (str(lower_bounds), str(higher_bounds)))

        # Execute the various intances of the computation
        logger.info("Initializing execution of data analysis")
        start_time = time.time()
        outputs = self._execute(records)
        logger.debug("Finished executing the computation: " + str(time.time() - start_time))

        
        new_epsilon = self._estimate_epsilon(outputs,
                                             self._bound_range(lower_bounds, higher_bounds),
                                             0.1)
        
        # Ensure output is within bounds
        for output in outputs:
            sanitize(output, lower_bounds, higher_bounds)

        # Ensure that the output dimension was the same for all
        # instances of the computation
        lengths = set([len(output) for output in outputs])
        lens_peek = lengths.pop()
        lengths.add(lens_peek)
        if len(lengths) > 1 or lens_peek != len(higher_bounds) or \
                lens_peek != len(lower_bounds):
            raise logger.exception("Output dimension is varying for each instance of the computation")

        # Aggregate and privatize the final output from various instances
        final_output = privatize(self.epsilon, lower_bounds,
                                 higher_bounds, outputs)
        return final_output

    @profile_func
    def _simple_get_data_bounds(self, records, epsilon):
        compute_driver = self.compute_driver_class()
        return compute_driver.get_output_bounds()

    @profile_func
    def _get_data_bounds(self, records, epsilon):
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

        hist = dpalgos.histogram(records_transpose, sensitive, epsilon)
        logger.debug("Ask compute driver what percentile to calculate")
        percentile_values = compute_driver.get_percentiles(hist)
        
        logger.debug("Estimating percentiles")
        lower_percentiles = []
        higher_percentiles = []
        for index in range(len(records_transpose)):
            if not sensitive[index]:
                lower_percentiles.append(0)
                higher_percentiles.append(0)
            else:
                lp = dpalgos.estimate_percentile(percentile_values[index][0],
                                                 records_transpose[index],
                                                 epsilon / (3 * len(records_transpose)),
                                                 min_vals[index],
                                                 max_vals[index])
                hp = dpalgos.estimate_percentile(percentile_values[index][1],
                                                 records_transpose[index],
                                                 epsilon / (3 * len(records_transpose)),
                                                 min_vals[index],
                                                 max_vals[index])
                lower_percentiles.append(lp)
                higher_percentiles.append(hp)

        logger.debug("Finished percentile estimation")
        logger.debug("Output bound estimation in progress")
        # Use the ComputeDriver's bound generator to generate the
        # output bounds
        return compute_driver.get_output_bounds(lower_percentiles,
                                                higher_percentiles)

    @profile_func
    def _get_data_bounds_parallel(self, records, epsilon):
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
        hist = dpalgos.histogram(records_transpose, sensitive, epsilon)
        logger.debug("Ask compute driver what percentile to calculate")
        percentile_values = compute_driver.get_percentiles(hist)

        logger.debug("Estimating percentiles in parallel")
        lower_percentiles = [0] * len(records_transpose)
        higher_percentiles = [0] * len(records_transpose)

        pipes = []
        procs = []
        for index in range(len(records_transpose)):
            if sensitive[index]:
                p, c = Pipe()
                proc = Process(target=spawn(dpalgos.estimate_percentile),
                               args=(c, percentile_values[index][0],
                                     records_transpose[index],
                                     epsilon / (3 * len(records_transpose)),
                                     min_vals[index],
                                     max_vals[index]))
                pipes.append((p, c,))
                procs.append(proc)
                proc.start()

                p, c = Pipe()
                proc = Process(target=spawn(dpalgos.estimate_percentile),
                               args=(c, percentile_values[index][1],
                                     records_transpose[index],
                                     epsilon / (3 * len(records_transpose)),
                                     min_vals[index],
                                     max_vals[index]))

                pipes.append((p, c,))
                procs.append(proc)
                proc.start()
            else:
                procs.append(None)
                procs.append(None)
                pipes.append(None)
                pipes.append(None)
                

        for index in range(len(records_transpose)):
            if sensitive[index]:
                procs[2 * index].join()
                lower_percentiles[index] = pipes[2 * index][0].recv()

                procs[2 * index + 1].join()
                higher_percentiles[index] = pipes[2 * index + 1][0].recv()

        logger.debug("Finished parallel percentile estimation")
        logger.debug("Output bound estimation in progress")
        # Use the ComputeDriver's bound generator to generate the
        # output bounds
        return compute_driver.get_output_bounds(lower_percentiles,
                                                higher_percentiles)
    
    @profile_func
    def _get_blocks(self, records):
        # TODO: Check if we can use random.sample instead of
        # shuffle. Heavy performance benefits.
        # random.shuffle(records)
        return self.blocker.get_blocks(records)
    
    def _apply_compute_driver(self, block):
        """
        Run the provided computation on the block of records
        """
        compute_driver = self.compute_driver_class()
        cur_output = GuptOutput()
        cur_output.append(compute_driver.initialize())
        for record in block:
            cur_output.append(compute_driver.execute(record))
        cur_output.extend(compute_driver.finalize())
        return cur_output

    @profile_func
    def _execute(self, records, mapper=map):
        """
        Execute the computation provider in a differentially private
        manner for the given set of records.
        """
        outputs = []
        blocks = self._get_blocks(records)
        
        # temp_blocks = blocks[:len(blocks) / 10]
        # all_data = []
        # for block in blocks:
        #     all_data.extend(block) 
        # est_outputs = mapper(self._apply_compute_driver, temp_blocks)
        # real_output = self._apply_compute_driver(all_data)
        # self.est_error = self._estimate_error(self._avg_multidim(est_outputs), real_output)
        # logger.info("Estimated estimation error is " + str(self.est_error))
        
        logger.debug("Starting data analytics on %d blocks" % (len(blocks)))
        return mapper(self._apply_compute_driver, blocks)

    # def _estimate_error(self, avg_output, real_output):
    #     """
    #     Estimate the average normalized difference between two sets of
    #     outputs
    #     """
    #     errors = []
    #     self._recur_estimate_error(avg_output, real_output, errors)
    #     return float(sum(errors)) / len(errors)

    # def _recur_estimate_error(self, avg_output, real_output, errors):
    #     if not isiterable(avg_output):
    #         errors.append(float(abs(avg_output - real_output)) / real_output)
    #         return

    #     if not isiterable(avg_output[0]):
    #         for index in range(len(avg_output)):
    #             self._recur_estimate_error(avg_output[index], real_output[index], errors)
    #         return

    #     for index in range(len(avg_output[0])):
    #         self._recur_estimate_error([cur_output[index] for cur_output in avg_output],
    #                                    [cur_output[index] for cur_output in real_output],
    #                                    errors)
    
    @profile_func
    def _parallel_execute(self, records):
        """
        Differentially private execution of the computation provider
        in a parallel fashion for the given set of records.
        """
        # Not using multiprocessing.Pool.map because it is having
        # issues with pickling of functions and various data
        # structures. 
        return self._execute(records, mapper=parmap)

    def _sanitize_multidim(self, record, lower_bounds, higher_bounds):
        if not isiterable(record):
            # TODO: Raise Exception
            logger.error("Sanitize function expects iterable objects")
            return
        
        if isiterable(record[0]): # Multidimensional output
            for index in range(len(record)):
                self._sanitize_multidim(record[index], lower_bounds[index], higher_bounds[index])
        else:
            for index in range(len(record)):
                if record[index] < lower_bounds[index]:
                    record[index] = lower_bounds[index]
                elif record[index] > higher_bounds[index]:
                    record[index] = higher_bounds[index]

    def _compute_sample_variance(self, est_ans):
        avg_ans = MultiDimensional.avg(est_ans)
        print 'Avg ANS', MultiDimensional.get_dimensionality(avg_ans)
        vals = [MultiDimensional.mul(MultiDimensional.sub(ans, avg_ans),
                                     MultiDimensional.sub(ans, avg_ans))
                for ans in est_ans]
        print len(vals), len(est_ans)
        return MultiDimensional.avg(vals)

    def _estimate_epsilon(self, outputs, range_bound, accuracy):
        non_private_outputs = outputs[:len(outputs) / 10]
        print 'No of non private outputs', len(non_private_outputs)
        est_sample_variance = self._compute_sample_variance(non_private_outputs)
        logging.info('Sample variance is' + str(est_sample_variance))
        print MultiDimensional.get_dimensionality(est_sample_variance)
        real_ans = MultiDimensional.avg(non_private_outputs)
        print MultiDimensional.get_dimensionality(real_ans)
        sd = MultiDimensional.mul_scalar(real_ans, accuracy)
        print MultiDimensional.get_dimensionality(sd)
        var = MultiDimensional.mul(sd, sd)
        print MultiDimensional.get_dimensionality(var)
        var_minus_sample_variance = MultiDimensional.sub(var, est_sample_variance)
        print MultiDimensional.get_dimensionality(var_minus_sample_variance)
        sqrt_var_minus_sample_variance = MultiDimensional.apply_to_each_scalar(var_minus_sample_variance,
                                                                               lambda x : math.sqrt(abs(x)))
        print MultiDimensional.get_dimensionality(sqrt_var_minus_sample_variance)

        range_bound = MultiDimensional.div_scalar(range_bound, len(outputs))
        epsilon = MultiDimensional.div(range_bound, sqrt_var_minus_sample_variance)
        epsilons = MultiDimensional.get_scalars(epsilon)
        epsilon = sum(epsilons)
        logger.info("Epsilon needed is %g, but given %g" % (epsilon, self.epsilon))

    def _bound_range(self, lower_bounds, higher_bounds):
        return MultiDimensional.abs(MultiDimensional.sub(lower_bounds, higher_bounds))

    def _perturb(self, bound_ranges, epsilon):
        if not isiterable(bound_ranges):
            return dpalgos.gen_noise(self.sensitivity_factor * float(bound_ranges) / epsilon)
        
        return [self._perturb(br, epsilon / len(bound_ranges)) for br in bound_ranges]

    @profile_func
    def _privatize(self, epsilon, lower_bounds, higher_bounds, outputs):
        """
        Converts the output of many instances of the computation
        into a differentially private answer
        """
        epsilon = epsilon / (3 * len(outputs[0]))
        bound_ranges = self._bound_range(lower_bounds, higher_bounds)

        final_output = MultiDimensional.avg(outputs)

        # Add a Laplacian noise in order to ensure differential privacy
        for index in range(len(final_output)):
            logger.info("Final Answer (Unperturbed) Dimension " + str(index) + " = " + str(final_output[index]))
            noise = self._perturb(bound_ranges[index], (epsilon * len(outputs)))
            logger.info("Perturbation = " + str(noise))
            final_output[index] = MultiDimensional.add(final_output[index], noise)
            logger.info("Final Answer (Perturbed) Dimension " + str(index) + " = " + str(final_output[index]))
        return final_output

    @profile_func
    def start(self):
        logger.info("Starting normal differentially private analysis")
        return self._start_diff_analysis(ret_bounds=self._get_data_bounds_parallel,
                                         sanitize=self._sanitize_multidim,
                                         privatize=self._privatize)

    @profile_func
    def start_windsorized(self):
        """
        Start the differentially private data analysis as defined by
        "Privacy-preserving Statistics Estimation with Optimal
        Convergence Rates" by Adam Smith, 2011
        """
        logger.info("Starting windsorized differentially private analysis")
        return self._start_diff_analysis(ret_bounds=self._simple_get_data_bounds,
                                         sanitize=lambda x, y, z: None,
                                         privatize=self._privatize_windsorized)

    @profile_func
    def start_nonprivate(self):
        logger.info("Starting non private analysis")
        return self._start_nonprivate_analysis()
    
if __name__ == '__main__':
    print >> sys.stderr, "This is a library and should not be executed standalone"
    sys.exit(1)
