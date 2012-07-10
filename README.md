## Introduction ##
GUPT is a privacy preserving data mining platform. GUPT guarantees
that the output of data analyses performed using this platform
guarantees privacy as defined by [differential privacy](http://en.wikipedia.org/wiki/Differential_privacy).
In order to use arbitrary binaries for data analysis in GUPT, you
will need to invoke the binary from the GuptComputeProvider object,
and possibly pipe IO through some form of IPC mechanism.

## Detailed Description ##
A detailed description of the GUPT architecture and mechanisms is
deferred to the [SIGMOD '12 paper](http://www.cs.berkeley.edu/~prmohan/papers/gupt.pdf).

## Advantages of GUPT ##
1. Enables differentially private data mining on black box programs
2. Parallel implementation ensures high performance. By virtue of the
   algorithm used internally, your program might run even faster than
   it does natively (outside of the GUPT Runtime).
3. Research platform with many hooks for evaluating various mechanisms
   for enabling Differential Privacy

## Instructions ##
1. You will need Python 2.7+ and the NumPy (http://numpy.scipy.org/)
library. Additionally you might also want to install SciPy
(http://www.scipy.org/) and scikit-learn (http://scikit-learn.org/stable/)
for performing additional machine learning.

2. Open the terminal and execute the following command replacing
GUPT-DIRECTORY with the path to where you downloaded GUPT to:
        export PYTHONPATH=GUPT-DIRECTORY:$PYTHONPATH

3. The samples folder lists various example scripts. Go into any of
those directories and execute the python script.

## Project Members ##
* Prashanth Mohan (http://www.cs.berkeley.edu/~prmohan)
* Abhradeep Guha Thakurta (https://sites.google.com/site/guhathakurtaabhradeep/)
* Elaine Shi (http://www.eecs.berkeley.edu/~elaines/)
* Dawn Song (http://www.cs.berkeley.edu/~dawnsong/)
* David Culler (http://www.cs.berkeley.edu/~culler/)

## Citations ##
When citing the project, please use this [bib file](http://www.cs.berkeley.edu/~prmohan/papers/gupt.bib)
