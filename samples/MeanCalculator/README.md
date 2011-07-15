This SampleProgram takes as input a stream of random numbers between 0
and 1 and finds the mean of the stream. In order to run this sample
program, you will first need to set the PYTHONPATH environment
variable to point to the GUPT platform. You can do this by the
following command:

$ export PYTHONPATH=<gupt-dir>:$PYTHONPATH

You should replace "<gupt-dir>" with the path to the directory which
contains gupt.py. You can now run the sample program by executing the
sensordb.py file.

Sample output:
$ python sensordb.py
  INFO root: Output bounds are [0.25694077215742833] and [0.2985709368406444]
  INFO root: Initializing execution of data analysis
  INFO root: Num Records: 10000, Block size: 256, Num blocks: 39
  INFO root: Final Answer (Unperturbed) Dimension 0 = 0.503764
  INFO root: Final Answer (Perturbed) Dimension 0 = 0.503997
[0.50399742275119286]

