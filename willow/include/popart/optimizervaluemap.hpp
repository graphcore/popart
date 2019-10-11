#ifndef GUARD_NEURALNET_OPTIMIZERVALUEMAP_HPP
#define GUARD_NEURALNET_OPTIMIZERVALUEMAP_HPP

#include <memory>
#include <popart/names.hpp>
#include <popart/optimizervalue.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/tensornames.hpp>

namespace popart {

class OptimizerValueMap {
public:
  OptimizerValueMap(OptimizerValue g) : defaultOptVal(g) {}

  // Return the OptimizerValue specific to "id" if there is one, otherwise
  // return the defaultOptVal OptimizerValue
  OptimizerValue get(const TensorId &id) const;

  // Register a specific OptimizerValue for a Tensor
  void insertSpecific(const TensorId &, OptimizerValue);

  bool hasSpecific(const TensorId &id) const {
    return specifics.find(id) != specifics.end();
  }

  OptimizerValue getDefault() const { return defaultOptVal; }

  // Check for compatibility of OptimizerValueMaps - can one replace another
  // after Graph construction without requiring changes to the compuatation
  // Graph?
  bool validReplacement(const OptimizerValueMap &rhs) const;

private:
  std::map<TensorId, OptimizerValue> specifics;

  // The fall-back for all Tensors without a specific OptimizerValue
  OptimizerValue defaultOptVal;
};

} // namespace popart

#endif
