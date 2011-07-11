import hashlib
import gupt
from computedriver.computedriver import GuptComputeDriver

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

    def get_output_bounds(self, first_quartile, third_quartile):
        return [first_quartile[2]], [third_quartile[2]]


if __name__ == '__main__':
    reader = art.get_reader()
    runtime = gupt.GuptRunTime(MeanComputer, reader, 1.0)
    output = runtime.start()
    print output
    
