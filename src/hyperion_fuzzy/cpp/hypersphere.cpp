#include "hypersphere.h"
#include <numeric>
#include <cstring>

Hypersphere::Hypersphere(const std::vector<double>& center, double radius,
                         const std::vector<std::vector<double>>& initial_elements)
    : center(center), radius(radius), initial_elements(initial_elements) {
    computeUx();
}

// New Constructor Using C Arrays
/*Hypersphere::Hypersphere(double* center, int center_size, 
    double* elements, int num_elements, int element_size, 
    double radius) 
    : center(center, center + center_size), radius(radius) {

    // Convert `double*` to `std::vector<std::vector<double>>`
    initial_elements.resize(num_elements, std::vector<double>(element_size));
    for (int i = 0; i < num_elements; ++i) {
        for (int j = 0; j < element_size; ++j) {
            initial_elements[i][j] = elements[i * element_size + j];
        }
    }

    computeUx();
}*/

void Hypersphere::setCenter(const std::vector<double>& new_center) {
    center = new_center;
}

std::vector<double> Hypersphere::getCenter() const {
    return center;
}

void Hypersphere::setRadius(double new_radius) {
    radius = new_radius;
}

double Hypersphere::getRadius() const {
    return radius;
}

void Hypersphere::computeUx() {
    int num_elements = initial_elements.size();
    int element_size = initial_elements[0].size();

    ux = std::vector<double>(element_size, 0.0);
    for (const auto& elem : initial_elements) {
        for (size_t i = 0; i < element_size; ++i) {
            ux[i] += elem[i];
        }
    }
    for (size_t i = 0; i < element_size; ++i) {
        ux[i] /= num_elements;
    }
}

std::vector<double> Hypersphere::getUx() const {
    return ux;
}

const std::vector<std::vector<double>>& Hypersphere::getInitialElements() const {
    return initial_elements;
}

const std::vector<std::tuple<std::vector<double>, int, double>>& Hypersphere::getAssignments() const {
    return assignments;
}

void Hypersphere::addAssignment(const std::vector<double>& array, int value, double weight) {
    assignments.emplace_back(array, value, weight);
}

void Hypersphere::clearAssignments() {
    assignments.clear();
}

// C-style API implementation
extern "C" {

    Hypersphere* create_hypersphere(double* center, int center_size, 
                                double* elements, int num_elements, int element_size, 
                                double radius) {
        // Convert 'center' from double* to std::vector<double>
        std::vector<double> center_vec(center, center + center_size);
        
        // Convert 'elements' from double* to std::vector<std::vector<double>>
        std::vector<std::vector<double>> elements_vec(num_elements, std::vector<double>(element_size));
        for (int i = 0; i < num_elements; ++i) {
            for (int j = 0; j < element_size; ++j) {
                elements_vec[i][j] = elements[i * element_size + j];
            }
        }

        // Create Hypersphere using the converted vectors
        return new Hypersphere(center_vec, radius, elements_vec);
    }

    void delete_hypersphere(Hypersphere* instance) {
        delete instance;
    }

    void set_center(Hypersphere* instance, double* new_center, int size) {
        std::vector<double> new_center_vec(new_center, new_center + size);
        instance->setCenter(new_center_vec);
    }

    void get_center(Hypersphere* instance, double* out_center) {
        std::vector<double> center = instance->getCenter();
        std::memcpy(out_center, center.data(), center.size() * sizeof(double));
    }

    void set_radius(Hypersphere* instance, double new_radius) {
        instance->setRadius(new_radius);
    }

    double get_radius(Hypersphere* instance) {
        return instance->getRadius();
    }

    void get_ux(Hypersphere* instance, double* out_ux) {
        std::vector<double> ux = instance->getUx();
        std::memcpy(out_ux, ux.data(), ux.size() * sizeof(double));
    }

    void add_assignment(Hypersphere* instance, double* array, int size, int value, double weight) {
        std::vector<double> array_vec(array, array + size);
        instance->addAssignment(array_vec, value, weight);
    }

    void clear_assignments(Hypersphere* instance) {
        instance->clearAssignments();
    }

    int get_num_initial_elements(Hypersphere* instance) {
        return instance->getInitialElements().size();
    }

    void get_initial_elements(Hypersphere* instance, double* out_elements) {
        const std::vector<std::vector<double>>& elements = instance->getInitialElements();
        int index = 0;
        for (const auto& row : elements) {
            std::memcpy(&out_elements[index], row.data(), row.size() * sizeof(double));
            index += row.size();
        }
    }

    int get_num_assignments(Hypersphere* instance) {
        return static_cast<int>(instance->getAssignments().size());
    }

    void get_assignments(Hypersphere* instance, double* out_arrays, int* out_values, double* out_weights) {
        const auto& assignments = instance->getAssignments();
        int index = 0;
        
        for (const auto& [array, value, weight] : assignments) {
            std::memcpy(&out_arrays[index], array.data(), array.size() * sizeof(double));
            out_values[index] = value;
            out_weights[index] = weight;
            index += array.size();
        }
    }

}