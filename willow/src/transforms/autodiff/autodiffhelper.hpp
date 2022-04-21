
// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_GRAD_GROWER_HPP
#define GUARD_NEURALNET_GRAD_GROWER_HPP

#include <functional>

#include <transforms/autodiff/autodiffirinterface.hpp>

namespace popart {

/**
 * Base class for helper classes that grow gradients for the autodiff
 * transform. These classes implement functionality that previously was
 * implemented as private members of popart::Ir. To facilitate this refactoring
 * some popart::Ir methods are wrapped as protected methods in this base class.
 *
 * Note that AutodiffHelper objects are designed to be short life-time objects
 * that persist no longer than the duration of some function. The lifetime of
 * the dependency must be guaranteed by the users of this class.
 */
class AutodiffHelper {
public:
  // Constructor.
  explicit AutodiffHelper(AutodiffIrInterface &dep);

protected:
  // Reference to ir.
  std::reference_wrapper<AutodiffIrInterface> dep;
};

} // namespace popart

#endif
