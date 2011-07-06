import hashlib
import gupt

from datadriver.csvdriver import CSVDriver
from computedriver.computedriver import GuptComputeDriver

# Data Provider Code
def ART_filter(records):
    if records and records[0].find('ART') != -1:
        return True
    return False

def ART_transformer(records):
    return [int(hashlib.sha1(records[0]).hexdigest(), 16)] + records[1:]

reader = CSVDriver(filter=ART_filter, transformer=ART_transformer, delimiter=' ')
reader.set_data_source('sensordb.txt')
reader.set_input_bounds([[0, 0], [0, 1]])
reader.set_sensitiveness([False, True])


# Compute Provider Code
class MeanComputer(gupt.GuptComputeDriver):
    def initialize(self):
        self.total = 0
        self.num_records = 0

    def execute(self, record):
        if not record or len(record) != 2:
            return

        self.total += record[1]
        self.num_records += 1

    def finalize(self):
        return self.total / self.num_records

    def get_output_bounds(self, first_quartile, third_quartile):
        return [first_quartile[1]], [third_quartile[1]]


if __name__ == '__main__':
    runtime = gupt.GuptRunTime(MeanComputer, reader, 1.0)
    output = runtime.start()
    print output
    
