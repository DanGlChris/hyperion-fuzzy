#include <cmath>
#include <algorithm>
#include <numeric>
#include <dlib/optimization.h>
#include "../include/optimize_hypersphere.h"

// Function to calculate the squared norm of the difference between two points
double squared_norm(const double* x, const double* x_prime, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; ++i) {
        double diff = x[i] - x_prime[i];
        sum += diff * diff;
    }
    return sum;
}

// Objective function for optimization
double objective(const dlib::matrix<double, 0, 1>& params, 
                 const Hypersphere& hypersphere, 
                 const std::vector<Hypersphere*>& other_hyperspheres, 
                 double c) {
    double radius = params(0);
    std::vector<double> center(params.size() - 1);
    for (int i = 0; i < center.size(); ++i) {
        center[i] = params(i + 1);
    }

    // Positive part of the objective function
    double pos_part = c * std::accumulate(hypersphere.getAssignments().begin(), hypersphere.getAssignments().end(), 0.0,
        [](double sum, const std::tuple<std::vector<double>, int, double>& assignment) {
            return sum + std::get<2>(assignment);
        });

    // Negative part of the objective function
    double neg_part = 0.0;
    int total_elements = 0;
    for (const auto& hs : other_hyperspheres) {
        neg_part += std::accumulate(hs->getAssignments().begin(), hs->getAssignments().end(), 0.0,
            [&](double sum, const std::tuple<std::vector<double>, int, double>& assignment) {
                const std::vector<double>& array = std::get<0>(assignment);
                return sum + std::sqrt(squared_norm(array.data(), center.data(), center.size()));
            });
        total_elements += 1;
    }

    if (total_elements > 0) {
        neg_part /= total_elements;
    }

    return radius * radius + pos_part - neg_part;
}

void optimize(Hypersphere* hypersphere,
              std::vector<Hypersphere*>& other_hyperspheres,
              double c,
              double learning_rate,
              int max_iterations,
              double tolerance,
              int dim) {
    dlib::matrix<double, 0, 1> initial_params(dim + 1);
    initial_params(0) = hypersphere->getRadius();
    std::vector<double> center = hypersphere->getCenter();
    for (int i = 0; i < dim; ++i) {
        initial_params(i + 1) = center[i];
    }

   auto objective_wrapper = [&](const dlib::matrix<double, 0, 1>& params) -> double {
        return objective(params, *hypersphere, other_hyperspheres, c);
    };

    dlib::find_min_using_approximate_derivatives(
        dlib::bfgs_search_strategy(),
        dlib::objective_delta_stop_strategy(tolerance).be_verbose(),
        objective_wrapper,
        initial_params,
        -1
    );

    hypersphere->setRadius(initial_params(0));
    std::vector<double> new_center(dim);
    for (int i = 0; i < dim; ++i) {
        new_center[i] = initial_params(i + 1);
    }
    hypersphere->setCenter(new_center);
}