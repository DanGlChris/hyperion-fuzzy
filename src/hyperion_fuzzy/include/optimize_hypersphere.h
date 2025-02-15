#ifndef OPTIMIZE_HYPERSPHERE_H
#define OPTIMIZE_HYPERSPHERE_H

#include <vector>
#include "hypersphere.h"

// Declare the optimize function
void optimize(Hypersphere* hypersphere,
              std::vector<Hypersphere*>& other_hyperspheres,
              double c,
              double learning_rate,
              int max_iterations,
              double tolerance,
              int dim);

#endif // OPTIMIZE_HYPERSPHERE_H