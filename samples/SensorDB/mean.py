import logging
import hashlib
import gupt
from computedriver.computedriver import GuptComputeDriver
import time
import art

# Compute Provider Code
class MeanComputer(GuptComputeDriver):
    def initialize(self):
        self.total = 0
        self.num_records = 0

    def execute(self, record):
        if not record or len(record) != 3:
            return

        self.total += record[2]
        self.num_records += 1

    def finalize(self):
        return self.total / self.num_records

    def get_output_bounds(self, first_quartile=None, third_quartile=None):
        if not first_quartile or not third_quartile:
            return [0], [100]
        return [first_quartile[2]], [third_quartile[2]]


if __name__ == '__main__':
    logging.info("Executing mean with bounded output")
    reader = art.get_reader()
    start_time = time.time()
    runtime = gupt.GuptRunTime(MeanComputer, reader, 1.0)
    print runtime.start(), (time.time() - start_time) * 1000

    logging.info("Executing mean in windsorized method")
    reader = art.get_reader()
    start_time = time.time()
    runtime = gupt.GuptRunTime(MeanComputer, reader, 1.0)
    print runtime.start_windsorized(), (time.time() - start_time) * 1000
    
