import logging
import hashlib
import gupt
from computedriver.computedriver import GuptComputeDriver
import time

from scipy.cluster.vq import vq, kmeans2, whiten
import scipy.spatial.distance as dist
import scipy.interpolate
import numpy as np
from operator import itemgetter
import komarixdatadriver

logger = logging.getLogger(__name__)

# Compute Provider Code
class KMeansComputer(GuptComputeDriver):
    def initialize(self):
        self.recs = []

    def kmeans_cluster(self, data, buckets):
        data = np.array(data)
        data = whiten(data)
        c, l = kmeans2(data, buckets, minit='random')
        return c

    def execute(self, record):
        self.recs.append(record)

    def finalize(self):
        return sorted(self.kmeans_cluster(self.recs, 3), key=itemgetter(1))

    def get_output_bounds(self, first_quartile=None, third_quartile=None):
        return [[-13.453099999999999, -6.4654999999999996,
                 -7.5628000000000002, -4.7937000000000003,
                 -6.9695999999999998, -5.0107999999999997,
                 -4.6077000000000004, -4.9134000000000002,
                 -5.1200000000000001, -3.4201000000000001]] * 3, \
                 [[-0.010748000000000001, 5.9762000000000004,
                   5.6764000000000001, 5.3849,
                   4.7241999999999997, 6.4660000000000002,
                   6.3295000000000003, 4.5263,
                   4.7115999999999998, 4.9504999999999999]] * 3


def run_expt(epsilon):
    for blocker in gupt.GuptRunTime.get_data_blockers():
        reader = komarixdatadriver.get_reader()
        runtime = gupt.GuptRunTime(KMeansComputer, reader, 1.0, blocker_name=blocker, blocker_args=2)
        print runtime.start()
        
        reader = komarixdatadriver.get_reader()
        runtime = gupt.GuptRunTime(KMeansComputer, reader, 1.0, blocker_name=blocker, blocker_args=2)
        print runtime.start_windsorized()
        

if __name__ == '__main__':
    for epsilon in range(1, 2):
        for index in range(1):
            run_expt(epsilon)
        
