#include "include/optimize_hypersphere.h"
#include <pybind11/stl.h>
#include <vector>

PYBIND11_MODULE(optimize_module, m) {
    m.doc() = "Optimization module for Hypersphere";

    m.def("optimize_hypersphere", &optimize_hypersphere,
          py::arg("hypersphere"), 
          py::arg("other_hyperspheres"), 
          py::arg("num_other_hyperspheres"), 
          py::arg("c1"), 
          py::arg("learning_rate"), 
          py::arg("max_iterations"), 
          py::arg("tolerance"), 
          py::arg("dim"));
}