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
        runtime = gupt.GuptRunTime(MeanComputer, reader, epsilon, blocker_name=blocker, blocker_args=gamma)
        print runtime.start()
        
        reader = censusdatadriver.get_reader()
        runtime = gupt.GuptRunTime(MeanComputer, reader, epsilon, blocker_name=blocker, blocker_args=gamma)
        print runtime.start_windsorized()
        

if __name__ == '__main__':
    for epsilon in range(1, 10):
        for gamma in range(2, 5, 2):
            for index in range(2):
                run_expt(epsilon, gamma)
        
