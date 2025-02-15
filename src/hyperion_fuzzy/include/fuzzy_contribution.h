#ifndef FUZZY_CONTRIBUTION_H
#define FUZZY_CONTRIBUTION_H

#include <vector>
#include "hypersphere.h"

// Function declarations
double squared_norm(const double* x, const double* x_prime, int dim);

double rbf_kernel(const double* x, const double* x_prime, double sigma, int dim);

double G(const double* x, const Hypersphere& hypersphere, int dim, double E);

double conformal_kernel(const double* x, const double* x_prime, const Hypersphere& hypersphere, double sigma, double E, int dim);

void fuzzy_contribution(
    const double* x,
    std::vector<Hypersphere>& positive_hyperspheres, std::vector<Hypersphere>& negative_hyperspheres,
    double gamma, double sigma, double E,
    int& assigned_class, double& contribution, int dim
);

void predict(
    const double* transformed_data, int num_samples, int dim,
    const std::vector<Hypersphere>& positive_hyperspheres,
    const std::vector<Hypersphere>& negative_hyperspheres,
    double sigma, int* predictions
);

#endif // FUZZY_CONTRIBUTION_H