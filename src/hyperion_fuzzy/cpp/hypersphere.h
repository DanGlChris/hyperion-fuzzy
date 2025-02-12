#ifndef HYPERSPHERE_H
#define HYPERSPHERE_H

#include <vector>
#include <tuple>

struct Hypersphere {
    double* initial_elements;
    double* center;
    const double* ux;
    double radius;
    int num_elements;
    std::vector<std::tuple<const double*, int, double>> assignments;
};

#endif // HYPERSPHERE_H