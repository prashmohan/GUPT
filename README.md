## Introduction ##
GUPT is a privacy preserving data mining platform. GUPT guarantees
that the output of data analyses performed using this platform
guarantees privacy as defined by [differential
privacy](http://en.wikipedia.org/wiki/Differential_privacy). When
using GUPT, in order to use arbitrary binaries for data analysis, you
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

## Project Members ##
* Prashanth Mohan (http://www.cs.berkeley.edu/~prmohan)
* Abhradeep Guha Thakurta (https://sites.google.com/site/guhathakurtaabhradeep/)
* Elaine Shi (http://www.eecs.berkeley.edu/~elaines/)
* Dawn Song (http://www.cs.berkeley.edu/~dawnsong/)
* David Culler (http://www.cs.berkeley.edu/~culler/)

## Citations ##
When citing the project, please use this [bib file](http://www.cs.berkeley.edu/~prmohan/papers/gupt.bib)
