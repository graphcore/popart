// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_OPTIMIZERVALUEMAP_HPP
#define GUARD_NEURALNET_OPTIMIZERVALUEMAP_HPP

#include <cstddef>
#include <functional>
#include <map>
#include <string>
#include <popart/names.hpp>
#include <popart/optimizervalue.hpp>

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

  bool hasSpecific() const { return specifics.size() > 0; }

  OptimizerValue getDefault() const { return defaultOptVal; }

  // Check for compatibility of OptimizerValueMaps - can one replace another
  // after Graph construction without requiring changes to the compuatation
  // Graph?
  void validReplacement(const OptimizerValueMap &rhs) const;

  const std::map<TensorId, OptimizerValue> &getSpecifics() const {
    return specifics;
  }

private:
  std::map<TensorId, OptimizerValue> specifics;

  // The fall-back for all Tensors without a specific OptimizerValue
  OptimizerValue defaultOptVal;
};

} // namespace popart

namespace std {
template <> struct hash<popart::OptimizerValueMap> {
  std::size_t operator()(const popart::OptimizerValueMap &vmap) const;
};
} // namespace std

namespace popart {
inline std::size_t hash_value(const OptimizerValueMap &vmap) {
  return std::hash<OptimizerValueMap>()(vmap);
}
} // namespace popart
#endif
