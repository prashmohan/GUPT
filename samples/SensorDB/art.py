import hashlib
from datadriver.sensedb import SensorDBDriver
import time

def ART_transformer(records):
    return [int(hashlib.sha1(records[0]).hexdigest(), 16),
            int(time.mktime(records[1].timetuple())),
            float(records[2])]

def get_reader():
    reader = SensorDBDriver(transformer=ART_transformer)
    reader.set_data_source('ART')
    reader.set_input_bounds([[0, 0], [0, 0], [0, 100]])
    reader.set_sensitiveness([False, False, True])
    return reader

