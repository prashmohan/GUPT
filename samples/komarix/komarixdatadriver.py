from datadriver.csvdriver import CSVDriver

def ART_transformer(records):
    return map(float, records[:-1])

def get_reader():
    reader = CSVDriver(transformer=ART_transformer)
    reader.set_data_source('ds1.10.csv')
    reader.set_input_bounds([[-13.4531, -0.010748],
                             [-6.4655, 5.9762],
                             [-7.5628, 5.6764],
                             [-4.7937, 5.3849],
                             [-6.9696, 4.7242],
                             [-5.0108, 6.466],
                             [-4.6077, 6.3295],
                             [-4.9134, 4.5263],
                             [-5.12, 4.7116],
                             [-3.4201, 4.9505]])
    reader.set_sensitiveness([True, True, True, True, True, True, True, True, True, True])
    return reader

