# feedforward-study-python
A simple study of feedforward neural nets in python

Discoveries:

    An adaptive learning parameter is required, since as the neural net converges on the minimum the error surface becomes increasingly jagged. The learning rate must diminish or training will perpetually overshoot. Decreasing the learning rate is equivalent to "zooming in" on the error surface.

    There are long stretches where the error seems to not be dimishing. Relatedly, the adaptive delta rapidly diminishes to below the representational capacity of python.

    In the case of XOR, The net achieves linear seperability on the classes without diminishing the error to zero (or near zero). The error signal here is simple mean squared distance.
