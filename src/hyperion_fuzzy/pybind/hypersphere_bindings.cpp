#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/hypersphere.h"

namespace py = pybind11;

PYBIND11_MODULE(hypersphere_module, m) {
    m.doc() = "Python bindings for Hypersphere class";

    py::class_<Hypersphere>(m, "Hypersphere")
        .def(py::init<const std::vector<double>&, double, const std::vector<std::vector<double>>&>(),
             py::arg("center"), py::arg("radius"), py::arg("initial_elements"))
        .def("set_center", &Hypersphere::setCenter, py::arg("new_center"))
        .def("get_center", &Hypersphere::getCenter)
        .def("set_radius", &Hypersphere::setRadius, py::arg("new_radius"))
        .def("get_radius", &Hypersphere::getRadius)
        .def("get_ux", &Hypersphere::getUx)
        .def("add_assignment", &Hypersphere::addAssignment,
             py::arg("array"), py::arg("value"), py::arg("weight"))
        .def("clear_assignments", &Hypersphere::clearAssignments)
        .def("get_initial_elements", &Hypersphere::getInitialElements)
        .def("get_assignments", &Hypersphere::getAssignments);
}