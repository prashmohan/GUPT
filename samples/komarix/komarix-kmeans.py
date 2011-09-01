import logging
import gupt
from computedriver.computedriver import GuptComputeDriver

from scipy.cluster.vq import kmeans2, whiten
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
        data = whiten(np.array(data))
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


def run_expt(epsilon, gamma):
    for blocker in gupt.GuptRunTime.get_data_blockers():
        reader = komarixdatadriver.get_reader()
        runtime = gupt.GuptRunTime(KMeansComputer, reader, epsilon, blocker_name=blocker, blocker_args=gamma)
        print runtime.start()
        
        reader = komarixdatadriver.get_reader()
        runtime = gupt.GuptRunTime(KMeansComputer, reader, epsilon, blocker_name=blocker, blocker_args=gamma)
        print runtime.start_windsorized()
        

if __name__ == '__main__':
    epsilon_vals = range(1, 10)
    gamma_vals = range(1, 5)
    REPETITIONS = 5

    for epsilon in epsilon_vals:
        for gamma in gamma_vals:
            for index in range(REPETITIONS):
                run_expt(epsilon, gamma)
