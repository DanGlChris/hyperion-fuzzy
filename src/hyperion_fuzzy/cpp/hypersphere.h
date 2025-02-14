#ifndef HYPERSPHERE_H
#define HYPERSPHERE_H

#include <vector>

#ifdef _WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

class DLL_EXPORT Hypersphere {
private:
    std::vector<std::vector<double>> initial_elements;
    std::vector<double> center;
    double radius;
    std::vector<std::tuple<std::vector<double>, int, double>> assignments;
    std::vector<double> ux;

    void computeUx(); // Utility function to compute ux

public:
    Hypersphere(const std::vector<double>& center, double radius, 
                const std::vector<std::vector<double>>& initial_elements);

    void setCenter(const std::vector<double>& new_center);
    std::vector<double> getCenter() const;

    void setRadius(double new_radius);
    double getRadius() const;

    std::vector<double> getUx() const;
    void addAssignment(const std::vector<double>& array, int value, double weight);
    std::vector<std::tuple<std::vector<double>, int, double>> getAssignments() const;
    void clearAssignments();  // New method to clear assignments
};

// C-style API for DLL export
extern "C" {
    DLL_EXPORT Hypersphere* create_hypersphere(double* center, int center_size, 
                                               double* elements, int num_elements, int element_size, 
                                               double radius);
    DLL_EXPORT void delete_hypersphere(Hypersphere* instance);
    DLL_EXPORT void set_center(Hypersphere* instance, double* new_center, int size);
    DLL_EXPORT void get_center(Hypersphere* instance, double* out_center);
    DLL_EXPORT void set_radius(Hypersphere* instance, double new_radius);
    DLL_EXPORT double get_radius(Hypersphere* instance);
    DLL_EXPORT void get_ux(Hypersphere* instance, double* out_ux);
    DLL_EXPORT void add_assignment(Hypersphere* instance, double* array, int size, int value, double weight);
    DLL_EXPORT void clear_assignments(Hypersphere* instance);
}

#endif // HYPERSPHERE_H