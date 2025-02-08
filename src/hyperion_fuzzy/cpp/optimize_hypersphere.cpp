#include <vector>
#include <cmath>
#include <algorithm>

// Objective function to optimize hypersphere radius and center
double objective(const std::vector<double>& params, const std::vector<std::vector<double>>& elements,
                 const std::vector<std::vector<double>>& other_hyperspheres, double c1) {
    double radius = params[0];
    std::vector<double> center(params.begin() + 1, params.end());

    // Positive part: sum of contributions
    double pos_part = 0.0;
    for (const auto& element : elements) {
        double distance = 0.0;
        for (size_t i = 0; i < center.size(); ++i) {
            distance += (element[i] - center[i]) * (element[i] - center[i]);
        }
        pos_part += std::sqrt(distance);
    }
    pos_part *= c1;

    // Negative part: penalize overlap with other hyperspheres
    double neg_part = 0.0;
    for (const auto& other_center : other_hyperspheres) {
        double distance = 0.0;
        for (size_t i = 0; i < center.size(); ++i) {
            distance += (center[i] - other_center[i]) * (center[i] - other_center[i]);
        }
        neg_part += std::sqrt(distance);
    }

    return radius * radius + pos_part - neg_part;
}

// Gradient-free optimization (simple gradient descent for demonstration purposes)
std::vector<double> optimize_hypersphere(const std::vector<double>& initial_params,
                                         const std::vector<std::vector<double>>& elements,
                                         const std::vector<std::vector<double>>& other_hyperspheres,
                                         double c1, double learning_rate, int max_iters) {
    std::vector<double> params = initial_params;

    for (int iter = 0; iter < max_iters; ++iter) {
        double current_obj = objective(params, elements, other_hyperspheres, c1);

        // Update radius (first parameter)
        params[0] -= learning_rate * (2 * params[0]);

        // Update center (remaining parameters)
        for (size_t i = 1; i < params.size(); ++i) {
            params[i] -= learning_rate * (params[i]);
        }

        // Break if the improvement is negligible
        if (std::abs(current_obj - objective(params, elements, other_hyperspheres, c1)) < 1e-6) {
            break;
        }
    }

    return params;
}