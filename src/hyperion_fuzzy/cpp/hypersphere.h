#ifndef HYPERSPHERE_H
#define HYPERSPHERE_H

#include <vector>
#include <tuple>

// Define the Hypersphere struct
struct Hypersphere {
    double* initial_elements;
    double* center;
    const double* ux;
    double radius;
    int num_elements;
    std::vector<std::tuple<double*, int, double>> assignments;

    // Constructor
    Hypersphere() : initial_elements(nullptr), center(nullptr), ux(nullptr), radius(0.0), num_elements(0) {}

    // Destructor (if applicable)
    ~Hypersphere() {
        delete[] initial_elements; // Free allocated memory
        delete[] center;           // Free allocated memory
        // Note: ux is const, assume it is managed outside this struct
    }
};

extern "C" {
    // Function to get the assignments data
    __declspec(dllexport) void __cdecl get_assignments(const Hypersphere* sphere, double** output, int* output_size);
}

#endif // HYPERSPHERE_H