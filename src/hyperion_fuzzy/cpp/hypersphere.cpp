#include "hypersphere.h"

extern "C" {
    __declspec(dllexport) void __cdecl get_assignments(const Hypersphere* sphere, double** output, int* output_size) {
        if (!sphere) return; // Safety check

        *output_size = static_cast<int>(sphere->assignments.size());
        *output = new double[*output_size];  // Allocate memory for Python to read

        for (int i = 0; i < *output_size; i++) {
            (*output)[i] = std::get<2>(sphere->assignments[i]);  // Extracting only the double value
        }
    }
}