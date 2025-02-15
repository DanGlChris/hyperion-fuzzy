#include "hypersphere.h"
#include <numeric>

Hypersphere::Hypersphere(const std::vector<double>& center, double radius,
                         const std::vector<std::vector<double>>& initial_elements)
    : center(center), radius(radius), initial_elements(initial_elements) {
    computeUx();
}

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
    if (num_elements == 0) {
        ux = std::vector<double>(center.size(), 0.0);
        return;
    }
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