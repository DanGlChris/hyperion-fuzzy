#include <vector>
#include <cmath>
#include <numeric>

// Function to calculate Euclidean distance between two points
double euclidean_distance(const std::vector<double>& x, const std::vector<double>& y) {
    double sum = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        sum += (x[i] - y[i]) * (x[i] - y[i]);
    }
    return std::sqrt(sum);
}

// Function to compute distances between a point and a set of points
std::vector<double> compute_distances(const std::vector<double>& x, const std::vector<std::vector<double>>& points) {
    std::vector<double> distances;
    for (const auto& p : points) {
        distances.push_back(euclidean_distance(x, p));
    }
    return distances;
}