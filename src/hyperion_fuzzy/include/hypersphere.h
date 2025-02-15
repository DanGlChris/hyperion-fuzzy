#ifndef HYPERSPHERE_H
#define HYPERSPHERE_H

#include <vector>
#include <tuple>

class Hypersphere {
private:
    std::vector<std::vector<double>> initial_elements;
    std::vector<double> center;
    double radius;
    std::vector<std::tuple<std::vector<double>, int, double>> assignments;
    std::vector<double> ux;

    void computeUx();

public:
    Hypersphere(const std::vector<double>& center, double radius, 
                const std::vector<std::vector<double>>& initial_elements);

    void setCenter(const std::vector<double>& new_center);
    std::vector<double> getCenter() const;

    void setRadius(double new_radius);
    double getRadius() const;

    std::vector<double> getUx() const;
    const std::vector<std::vector<double>>& getInitialElements() const;
    const std::vector<std::tuple<std::vector<double>, int, double>>& getAssignments() const;

    void addAssignment(const std::vector<double>& array, int value, double weight);
    void clearAssignments();
};
#endif // HYPERSPHERE_H