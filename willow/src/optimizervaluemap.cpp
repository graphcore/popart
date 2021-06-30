// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/optimizervaluemap.hpp>

#include <boost/functional/hash.hpp>

namespace popart {

void OptimizerValueMap::insertSpecific(const TensorId &id, OptimizerValue ov) {
  auto found = specifics.find(id);
  if (found != specifics.end()) {
    std::ostringstream oss;
    oss << "Attempt to insert specific value for optimization Tensor " << id
        << "failed as there is already a specific value for " << id
        << " present. Bailing, in case this is an error.";
    throw error(oss.str());
  }
  specifics.insert({id, ov});
}

OptimizerValue OptimizerValueMap::get(const TensorId &id) const {
  auto found = specifics.find(id);
  if (found != specifics.end()) {
    return found->second;
  }
  return defaultOptVal;
}

void OptimizerValueMap::validReplacement(const OptimizerValueMap &ovm) const {

  try {
    defaultOptVal.validReplacement(ovm.defaultOptVal);
  } catch (error &err) {
    throw error("Default value is not a valid replacement. {}", err.what());
  }

  if (specifics.size() != ovm.specifics.size()) {
    throw error("Replacement OptimizerValueMap has a different number of "
                "specific tensors.");
  }

  for (const auto &id_ov : specifics) {
    const TensorId &id = id_ov.first;
    const auto &ov     = id_ov.second;
    auto ovm_found     = ovm.specifics.find(id);
    if (ovm_found == ovm.specifics.end()) {
      throw error("Could not find tensor {} in replacement OptimizerValueMap.",
                  id);
    }
    try {
      ov.validReplacement(ovm_found->second);
    } catch (error &err) {
      throw error("OptimizerValue for tensor {} is not a valid replacement.{}",
                  id,
                  err.what());
    }
  }
}

} // namespace popart

namespace std {
std::size_t std::hash<popart::OptimizerValueMap>::operator()(
    const popart::OptimizerValueMap &vmap) const {
  std::size_t seed = 0;
  boost::hash_combine(seed, vmap.getDefault());

  for (const auto &kv : vmap.getSpecifics()) {
    boost::hash_combine(seed, kv.first);
    boost::hash_combine(seed, kv.second);
  }
  return seed;
}
} // namespace std
