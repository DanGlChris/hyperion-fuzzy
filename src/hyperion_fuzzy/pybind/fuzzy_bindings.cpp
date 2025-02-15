#include "../include/fuzzy_contribution.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(fuzzy_contribution_module, m) {
    m.doc() = "Fuzzy contribution module";

    // Wrap the fuzzy_contribution function
    m.def("fuzzy_contribution", [](const py::array_t<double>& x,
                                   std::vector<Hypersphere>& positive_hyperspheres,
                                   std::vector<Hypersphere>& negative_hyperspheres,
                                   double gamma, double sigma, double E) {
        py::buffer_info x_buf = x.request();
        if (x_buf.ndim != 1) {
            throw std::runtime_error("x must be a 1D array");
        }
        int dim = x_buf.shape[0];
        const double* x_ptr = static_cast<double*>(x_buf.ptr);

        int assigned_class;
        double contribution;

        fuzzy_contribution(x_ptr, positive_hyperspheres, negative_hyperspheres,
                          gamma, sigma, E, assigned_class, contribution, dim);

        return py::make_tuple(assigned_class, contribution);
    }, py::arg("x"), py::arg("positive_hyperspheres"), py::arg("negative_hyperspheres"),
       py::arg("gamma"), py::arg("sigma"), py::arg("E"),
       "This function assigns a class based on fuzzy contribution.");

    // Wrap the predict function
    m.def("predict", [](const py::array_t<double>& transformed_data,
                        const std::vector<Hypersphere>& positive_hyperspheres,
                        const std::vector<Hypersphere>& negative_hyperspheres,
                        double sigma) {
        py::buffer_info data_buf = transformed_data.request();
        if (data_buf.ndim != 2) {
            throw std::runtime_error("transformed_data must be a 2D array");
        }
        int num_samples = data_buf.shape[0];
        int dim = data_buf.shape[1];
        const double* data_ptr = static_cast<double*>(data_buf.ptr);

        std::vector<int> predictions(num_samples);
        predict(data_ptr, num_samples, dim, positive_hyperspheres, negative_hyperspheres, sigma, predictions.data());

        return predictions;
    }, py::arg("transformed_data"), py::arg("positive_hyperspheres"), py::arg("negative_hyperspheres"),
       py::arg("sigma"),
       "This function predicts classes for transformed data.");
}