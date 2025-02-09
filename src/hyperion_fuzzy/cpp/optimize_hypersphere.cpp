#include <cmath>
#include <vector>
#include <algorithm>
#include <dlib/optimization.h>

struct Hypersphere {
    double* initial_elements;
    double* center;
    const double* ux;
    double radius;
    int num_elements;
    std::vector<std::tuple<const double*, int, double>> assignments;
};

double squared_norm(const double* x, const double* x_prime, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; i++) {
        double diff = x[i] - x_prime[i];
        sum += diff * diff;
    }
    return sum;
}

double objective(const dlib::matrix<double, 0, 1>& params, const Hypersphere& hypersphere, const std::vector<Hypersphere>& other_hyperspheres, double c1, int dim) {
    double radius = params(0);
    const double* center = &params(1);

    double pos_part = c1 * std::accumulate(hypersphere.assignments.begin(), hypersphere.assignments.end(), 0.0,
        [](double sum, const std::tuple<const double*, int, double>& assignment) {
            return sum + std::get<2>(assignment);
        });

    double neg_part = 0.0;
    int total_elements = 0; // To count total number of elements across all hyperspheres

    // Iterate over each hypersphere
    for (const auto& hs : other_hyperspheres) {
        // Use std::accumulate to sum the squared_norm of the first elements of the assignments
        neg_part += std::accumulate(hs.assignments.begin(), hs.assignments.end(), 0.0,
            [&](double sum, const std::tuple<const double*, int, double>& assignment) {
                return sum + squared_norm(std::get<0>(assignment), center, dim);
            });
        total_elements += 1; // Update the total count of elements
    }

    // Calculate the average
    if (total_elements > 0) {
        neg_part /= total_elements; // Average the squared distances
    }

    return radius * radius + pos_part - neg_part;
}

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
        dlib::matrix<double, 0, 1> initial_params(dim + 1);
        initial_params(0) = hypersphere.radius;
        for (int i = 0; i < dim; ++i) {
            initial_params(i + 1) = hypersphere.center[i];
        }

        auto objective_wrapper = [&](const dlib::matrix<double, 0, 1>& params) -> double {
            return objective(params, hypersphere, other_hyperspheres, c1, dim);
        };

        dlib::find_min_using_approximate_derivatives(
            dlib::bfgs_search_strategy(),
            dlib::objective_delta_stop_strategy(tolerance).be_verbose(),
            objective_wrapper,
            initial_params,
            -1
        );

        hypersphere.radius = initial_params(0);
        for (int i = 0; i < dim; ++i) {
            hypersphere.center[i] = initial_params(i + 1);
        }
    }
}