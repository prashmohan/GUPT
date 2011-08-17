import logging
import hashlib
import gupt
from computedriver.computedriver import GuptComputeDriver
import time
import censusdatadriver

logger = logging.getLogger(__name__)

# Compute Provider Code
class MeanComputer(GuptComputeDriver):
    def initialize(self):
        self.total_age = 0.0
        self.num_records = 0
        self.total_income = 0.0

    def execute(self, record):
        if not record:
            return

        self.total_age += record[0]
        self.num_records += 1

    def finalize(self):
        try:
            return self.total_age / self.num_records
        except Exception, e:
            logger.exception(e)
            print self.total_age, self.num_records
            return 0,0

    def get_output_bounds(self, first_quartile=None, third_quartile=None):
        if not first_quartile or not third_quartile:
            # return [0.0, 0.0], [150.0, 500000.0]
            return [0.0], [150.0]
        return first_quartile, third_quartile

def run_expt(epsilon, gamma):
    for blocker in gupt.GuptRunTime.get_data_blockers():
        reader = censusdatadriver.get_reader()
        runtime = gupt.GuptRunTime(MeanComputer, reader, float(epsilon), blocker_name=blocker, blocker_args=gamma)
        print runtime.start()
        
        reader = censusdatadriver.get_reader()
        runtime = gupt.GuptRunTime(MeanComputer, reader, float(epsilon), blocker_name=blocker, blocker_args=gamma)
        print runtime.start_windsorized()
        

if __name__ == '__main__':
    epsilon_vals = [1, 5]
    gamma_vals = range(1, 10)
    REPETITIONS = 20

    for epsilon in epsilon_vals:
        for gamma in gamma_vals:
            for index in range(REPETITIONS):
                run_expt(epsilon, gamma)

    reader = censusdatadriver.get_reader()
    runtime = gupt.GuptRunTime(MeanComputer, reader, 1.0)
    print runtime.start_nonprivate()
    del runtime
    del reader
