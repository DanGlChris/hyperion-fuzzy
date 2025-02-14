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
double G(const double* x, const Hypersphere& hypersphere, int dim, double E) {
    const std::vector<std::vector<double>>& initial_elements = hypersphere.getInitialElements();
    const std::vector<double>& ux = hypersphere.getUx();
    int num_elements = static_cast<int>(initial_elements.size());

    double sum = 0.0;
    for (int i = 0; i < num_elements; i++) {
        double distance2 = 0.0;
        double ux_distance2 = 0.0;
        for (int j = 0; j < dim; j++) {
            double diff = initial_elements[i][j] - x[j];
            distance2 += diff * diff;
            double ux_diff = ux[j] - initial_elements[i][j];
            ux_distance2 += ux_diff * ux_diff;
        }
        sum += std::exp(-distance2 / (ux_distance2 + E));
    }
    return sum;
}

// Conformal Kernel Function
double conformal_kernel(const double* x, const double* x_prime, const Hypersphere& hypersphere, double sigma, double E, int dim) {
    double G_x = G(x, hypersphere, dim, E);
    double G_x_prime = G(x_prime, hypersphere, dim, E);
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
        int assigned_hypersphere_p = -1;
        int assigned_hypersphere_n = -1;

        // Compute conformal kernel for positive hyperspheres
        for (int i = 0; i < num_positive; ++i) {
            double k = conformal_kernel(x, positive_hyperspheres[i].getCenter().data(), 
                                        positive_hyperspheres[i], sigma, E, dim);
            if (k < min_positive) {
                min_positive = k;
                assigned_hypersphere_p = i;
            }
        }

        // Compute conformal kernel for negative hyperspheres
        for (int i = 0; i < num_negative; ++i) {
            double k = conformal_kernel(x, negative_hyperspheres[i].getCenter().data(), 
                                        negative_hyperspheres[i], sigma, E, dim);
            if (k < min_negative) {
                min_negative = k;
                assigned_hypersphere_n = i;
            }
        }

        if (min_positive < min_negative) {
            const Hypersphere& neg_sphere = negative_hyperspheres[assigned_hypersphere_n];
            double d_to_other_boundary = std::abs(min_negative - neg_sphere.getRadius());
            double c_to_cen = 1 - 1 / std::sqrt(min_positive + gamma);
            double c_to_boundary = 1 - 1 / std::sqrt(d_to_other_boundary + gamma);
            *contribution = std::max(c_to_cen, c_to_boundary);
            *assigned_class = 1;
            positive_hyperspheres[assigned_hypersphere_p].addAssignment(
                std::vector<double>(x, x + dim), 1, *contribution
            );

        } else if (min_positive > min_negative) {
            const Hypersphere& ps_sphere = positive_hyperspheres[assigned_hypersphere_p];
            double d_to_other_boundary = std::abs(min_positive - ps_sphere.getRadius());
            double c_to_cen = 1 - 1 / std::sqrt(min_negative + gamma);
            double c_to_boundary = 1 - 1 / std::sqrt(d_to_other_boundary + gamma);
            *contribution = std::max(c_to_cen, c_to_boundary);
            *assigned_class = -1;
            negative_hyperspheres[assigned_hypersphere_n].addAssignment(
                std::vector<double>(x, x + dim), -1, *contribution
            );
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
                memberships_p[j] = conformal_kernel(x, positive_hyperspheres[j].getCenter().data(), positive_hyperspheres[j], sigma, 0.0, dim);
            }

            for (int j = 0; j < num_negative; ++j) {
                memberships_n[j] = conformal_kernel(x, negative_hyperspheres[j].getCenter().data(), negative_hyperspheres[j], sigma, 0.0, dim);
            }

            double max_membership_p = *std::min_element(memberships_p.begin(), memberships_p.end());
            double max_membership_n = *std::min_element(memberships_n.begin(), memberships_n.end());

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