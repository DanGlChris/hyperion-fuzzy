#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>
#include "hypersphere.h"

// Compute squared Euclidean distance
double squared_norm(const double* x, const double* x_prime, int dim) {
    double sum = 0.0;
    for (int i = 0; i < dim; i++) {
        double diff = x[i] - x_prime[i];
        sum += diff * diff;
    }
    return sum;
}

// Radial Basis Function (RBF) Kernel
double rbf_kernel(const double* x, const double* x_prime, double sigma, int dim) {
    double squared_dist = squared_norm(x, x_prime, dim);
    return std::exp(-squared_dist / (2 * sigma * sigma));
}

// Compute the conformal factor G(x)
double G(const double* x, const double* initial_elements, const double* ux, int num_elements, int dim, double E) {
    double sum = 0.0;
    for (int i = 0; i < num_elements; i++) {
        double distance2 = 0.0;
        double ux_distance2 = 0.0;
        for (int j = 0; j < dim; j++) {
            double diff = initial_elements[i * dim + j] - x[j];
            distance2 += diff * diff;
            double ux_diff = ux[i * dim + j] - initial_elements[i * dim + j];
            ux_distance2 += ux_diff * ux_diff;
        }
        sum += std::exp(-distance2 / (ux_distance2 + E));
    }
    return sum;
}

// Conformal Kernel Function
double conformal_kernel(const double* x, const double* x_prime, const Hypersphere& hypersphere, double sigma, double E, int dim) {
    double G_x = G(x, hypersphere.initial_elements, hypersphere.ux, hypersphere.num_elements, dim, E);
    double G_x_prime = G(x_prime, hypersphere.initial_elements, hypersphere.ux, hypersphere.num_elements, dim, E);
    double rbf = rbf_kernel(x, x_prime, sigma, dim);
    return G_x * rbf * G_x_prime;
}

// Fuzzy Contribution Function
extern "C" {
    __declspec(dllexport) void __cdecl fuzzy_contribution(
        const double* x,
        Hypersphere* positive_hyperspheres, Hypersphere* negative_hyperspheres,
        int num_positive, int num_negative, int dim,
        double gamma, double sigma, double E,
        int* assigned_class, double* contribution
    ) {
        double min_positive = std::numeric_limits<double>::infinity();
        double min_negative = std::numeric_limits<double>::infinity();
        int assigned_hypersphere_p = 0;
        int assigned_hypersphere_n = 0;

        // Compute conformal kernel for positive hyperspheres
        for (int i = 0; i < num_positive; ++i) {
            double k = conformal_kernel(x, positive_hyperspheres[i].center, positive_hyperspheres[i], sigma, E, dim);
            if (k < min_positive) {
                min_positive = k;
                assigned_hypersphere_p = i;
            }
        }

        // Compute conformal kernel for negative hyperspheres
        for (int i = 0; i < num_negative; ++i) {
            double k = conformal_kernel(x, negative_hyperspheres[i].center, negative_hyperspheres[i], sigma, E, dim);
            if (k < min_negative) {
                min_negative = k;
                assigned_hypersphere_n = i;
            }
        }

        if (min_positive < min_negative) {
            *contribution = 1.0 - (1.0 / std::sqrt(min_positive + gamma));
            *assigned_class = 1;
        } else if (min_positive > min_negative) {
            *contribution = 1.0 - (1.0 / std::sqrt(min_negative + gamma));
            *assigned_class = -1;
        } else {
            *contribution = 1.0;
            *assigned_class = 0;
        }
    }

    // Prediction Function
    __declspec(dllexport) void __cdecl predict(
        const double* transformed_data, int num_samples, int dim,
        const Hypersphere* positive_hyperspheres, int num_positive,
        const Hypersphere* negative_hyperspheres, int num_negative,
        double sigma, int* predictions
    ) {
        for (int i = 0; i < num_samples; ++i) {
            const double* x = &transformed_data[i * dim];

            std::vector<double> memberships_p(num_positive);
            std::vector<double> memberships_n(num_negative);

            for (int j = 0; j < num_positive; ++j) {
                memberships_p[j] = conformal_kernel(x, positive_hyperspheres[j].center, positive_hyperspheres[j], sigma, 0.0, dim);
            }

            for (int j = 0; j < num_negative; ++j) {
                memberships_n[j] = conformal_kernel(x, negative_hyperspheres[j].center, negative_hyperspheres[j], sigma, 0.0, dim);
            }

            double max_membership_p = *std::max_element(memberships_p.begin(), memberships_p.end());
            double max_membership_n = *std::max_element(memberships_n.begin(), memberships_n.end());

            if (max_membership_p > max_membership_n) {
                predictions[i] = 1;
            } else if (max_membership_n > max_membership_p) {
                predictions[i] = -1;
            } else {
                predictions[i] = 0;
            }
        }
    }
}