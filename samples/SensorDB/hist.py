import hashlib
import gupt
from computedriver.computedriver import GuptComputeDriver

import art

reader = art.get_reader()

# Compute Provider Code
class HistogramComputer(GuptComputeDriver):
    def initialize(self):
        self.buckets = [0] * 11
        self.num_records = 0

    def execute(self, record):
        if not record or len(record) != 3:
            return
        self.num_records += 1
        temp = record[1] / 10

        if temp > 10:
            temp = 10

        self.buckets[temp] += 1

    def finalize(self):
        return [[float(val) / self.num_records for val in self.buckets]]

    def get_output_bounds(self, first_quartile, third_quartile):
        return [0] * 11, [1] * 11

        
if __name__ == '__main__':
    runtime = gupt.GuptRunTime(HistogramComputer, reader, 1.0)
    output = runtime.start()
    print output
    
