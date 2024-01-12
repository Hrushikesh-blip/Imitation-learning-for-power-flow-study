# Imitation-learning-for-power-flow-study
Imitation learning for power flow study using Graph Neural Networks
Power engineering relies heavily on power flow analysis as a primary method
for arranging and controlling the operation of power systems. The issue of
common power flow can be broken down into a series of nonlinear equations,
typically solved through the use of numerical optimization. Methods of calculation such as the Newton-Raphson method. These methods, on the other
hand, may become computationally expensive when applied to large systems,
and convergence to the global optimum is not guaranteed. In recent years,
a number of strategies Graph Neural Networks have been proposed with the
intention of accelerating the computation of the power flow solutions without significantly compromising the accuracy of the results. This collection
of models is able to learn properties independently of a larger graph’s structure. Consequently, given that power systems are typically represented as
graphs, these methods can, in theory, be generalized to systems of varying
sizes and topologies. On the other hand, the majority of the approaches that
are currently being used have only been tested on systems that have a single
topology, and none of them have been trained simultaneously on systems
with varying topologies.
The study presented in this project allowed to draw important insights about
the applicability of GNN as power flow solvers.
