## Introduction ##
GUPT is a privacy preserving data mining platform. GUPT guarantees
privacy in statistical databases using Differential Privacy. In order
to use arbitrary binaries, for your computation, you will need to
invoke your binary from the GuptComputeProvider object, and possibly
pipe IO through some form of IPC mechanism.

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
