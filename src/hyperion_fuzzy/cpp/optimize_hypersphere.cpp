#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric> // Include to use std::accumulate
#include <dlib/optimization.h>
#include "hypersphere.h"

// Function to calculate the squared norm of the difference between two points
double squared_norm(const double* x, const double* x_prime, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; i++) {
        double diff = x[i] - x_prime[i];
        sum += diff * diff;
    }
    return sum;
}

// Objective function for optimization
double objective(const dlib::matrix<double, 0, 1>& params, 
                 const Hypersphere& hypersphere, 
                 const std::vector<Hypersphere>& other_hyperspheres, 
                 double c1, 
                 int dim) {
    double radius = params(0);
    const double* center = &params(1);

    // Positive part of the objective function
    double pos_part = c1 * std::accumulate(hypersphere.assignments.begin(), hypersphere.assignments.end(), 0.0,
        [](double sum, const std::tuple<const double*, int, double>& assignment) {
            return sum + std::get<2>(assignment);
        });

    // Negative part of the objective function
    double neg_part = 0.0;
    int total_elements = 0; // To count total number of elements across all hyperspheres

    // Iterate over each hypersphere
    for (const auto& hs : other_hyperspheres) {
        // Sum the squared norms of the distances from the center
        neg_part += std::accumulate(hs.assignments.begin(), hs.assignments.end(), 0.0,
            [&](double sum, const std::tuple<const double*, int, double>& assignment) {
                return sum + squared_norm(std::get<0>(assignment), center, dim);
            });
        total_elements += 1; // Update the total count of elements
    }

    // Calculate the average for the negative part
    if (total_elements > 0) {
        neg_part /= total_elements; // Average the squared distances
    }

    // Return the objective value
    return radius * radius + pos_part - neg_part;
}

// Exported function to optimize hypersphere parameters
extern "C" {
    __declspec(dllexport) void optimize_hypersphere(
        Hypersphere& hypersphere,
        const std::vector<Hypersphere>& other_hyperspheres,
        double c1,
        double learning_rate,
        int max_iterations,
        double tolerance,
        int dim
    ) {
        // Initialize parameters for optimization
        dlib::matrix<double, 0, 1> initial_params(dim + 1);
        initial_params(0) = hypersphere.radius; // Set initial radius
        for (int i = 0; i < dim; ++i) {
            initial_params(i + 1) = hypersphere.center[i]; // Set initial center
        }

        // Wrapper for the objective function
        auto objective_wrapper = [&](const dlib::matrix<double, 0, 1>& params) -> double {
            return objective(params, hypersphere, other_hyperspheres, c1, dim);
        };

        // Perform optimization using Dlib
        dlib::find_min_using_approximate_derivatives(
            dlib::bfgs_search_strategy(),
            dlib::objective_delta_stop_strategy(tolerance).be_verbose(),
            objective_wrapper,
            initial_params,
            -1
        );

        // Update the hypersphere parameters after optimization
        hypersphere.radius = initial_params(0);
        for (int i = 0; i < dim; ++i) {
            hypersphere.center[i] = initial_params(i + 1);
        }
    }
}