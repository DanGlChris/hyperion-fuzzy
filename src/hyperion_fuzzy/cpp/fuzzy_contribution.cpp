#include <vector>
#include <cmath>
#include <algorithm>

// Function to calculate Euclidean distance
double euclidean_distance(const std::vector<double>& x, const std::vector<double>& y) {
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        sum += (x[i] - y[i]) * (x[i] - y[i]);
    }
    return std::sqrt(sum);
}

// Function to calculate the RBF kernel
double rbf_kernel(const std::vector<double>& x, const std::vector<double>& x_prime, double sigma) {
    double distance_squared = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        distance_squared += (x[i] - x_prime[i]) * (x[i] - x_prime[i]);
    }
    return std::exp(-distance_squared / (2 * sigma * sigma));
}

// Function to compute G(x, hypersphere)
double G(const std::vector<double>& x, const std::vector<std::vector<double>>& initial_elements, 
         const std::vector<double>& ux, double epsilon) {
    double sum = 0.0;
    for (const auto& element : initial_elements) {
        double distance = euclidean_distance(element, x);
        double ux_distance = euclidean_distance(ux, element);
        sum += std::exp(-distance * distance / (ux_distance * ux_distance + epsilon));
    }
    return sum;
}

// Function to compute conformal kernel
double conformal_kernel(const std::vector<double>& x, const std::vector<double>& x_prime, 
                        const std::vector<std::vector<double>>& initial_elements, 
                        const std::vector<double>& ux, double sigma, double epsilon) {
    double g_x = G(x, initial_elements, ux, epsilon);
    double g_x_prime = G(x_prime, initial_elements, ux, epsilon);
    double rbf = rbf_kernel(x, x_prime, sigma);
    return g_x * rbf * g_x_prime;
}

// Fuzzy contribution function
void fuzzy_contribution(const std::vector<double>& x, 
                        const std::vector<std::vector<double>>& positive_centers, 
                        const std::vector<std::vector<double>>& negative_centers, 
                        const std::vector<std::vector<std::vector<double>>>& positive_elements, 
                        const std::vector<std::vector<std::vector<double>>>& negative_elements, 
                        double c1, double sigma, double epsilon, 
                        int& label, double& contribution) {
    double min_value_p = std::numeric_limits<double>::infinity();
    double min_value_n = std::numeric_limits<double>::infinity();

    // Positive hypersphere membership
    for (size_t i = 0; i < positive_centers.size(); ++i) {
        const auto& center = positive_centers[i];
        const auto& elements = positive_elements[i];
        std::vector<double> ux(center.size(), 0.0);

        // Compute ux for the current hypersphere
        for (const auto& el : elements) {
            for (size_t j = 0; j < el.size(); ++j) {
                ux[j] += el[j];
            }
        }
        for (size_t j = 0; j < ux.size(); ++j) {
            ux[j] /= elements.size();
        }

        double kernel_value = conformal_kernel(x, center, elements, ux, sigma, epsilon);
        if (kernel_value < min_value_p) {
            min_value_p = kernel_value;
        }
    }

    // Negative hypersphere membership
    for (size_t i = 0; i < negative_centers.size(); ++i) {
        const auto& center = negative_centers[i];
        const auto& elements = negative_elements[i];
        std::vector<double> ux(center.size(), 0.0);

        for (const auto& el : elements) {
            for (size_t j = 0; j < el.size(); ++j) {
                ux[j] += el[j];
            }
        }
        for (size_t j = 0; j < ux.size(); ++j) {
            ux[j] /= elements.size();
        }

        double kernel_value = conformal_kernel(x, center, elements, ux, sigma, epsilon);
        if (kernel_value < min_value_n) {
            min_value_n = kernel_value;
        }
    }

    // Assign label based on minimum values
    if (min_value_p < min_value_n) {
        label = 1;
        contribution = 1.0 - 1.0 / std::sqrt(min_value_p * min_value_p + c1);
    } else {
        label = -1;
        contribution = 1.0 - 1.0 / std::sqrt(min_value_n * min_value_n + c1);
    }
}