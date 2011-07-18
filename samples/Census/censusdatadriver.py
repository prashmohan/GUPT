from datadriver.csvdriver import CSVDriver

def ART_transformer(records):
    """Return (age, gross income)"""
    return [float(records[0])]

def get_reader():
    reader = CSVDriver(transformer=ART_transformer)
    reader.set_data_source('census-income.data')
    # reader.set_input_bounds([[0.0, 200.0], [0.0, 50000.0]])
    reader.set_input_bounds([[0.0, 200.0]])
    reader.set_sensitiveness([True])
    return reader

