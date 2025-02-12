#include <cmath>
#include <stdexcept>

extern "C" {
    __declspec(dllexport) double __cdecl euclideanDistance(const double* point1, const double* point2, size_t size) {
        if (size == 0) {
            return -1.0; // Indicate an error
        }

        double sum = 0.0;
        for (size_t i = 0; i < size; ++i) {
            sum += std::pow(point1[i] - point2[i], 2);
        }

        return std::sqrt(sum);
    }
}
