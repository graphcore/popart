// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/optimizervaluemap.hpp>

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
};

bool OptimizerValueMap::validReplacement(const OptimizerValueMap &ovm) const {

  if (!defaultOptVal.validReplacement(ovm.defaultOptVal)) {
    return false;
  }

  if (specifics.size() != ovm.specifics.size()) {
    return false;
  }

  for (const auto &id_ov : specifics) {
    const TensorId &id = id_ov.first;
    const auto &ov     = id_ov.second;
    auto ovm_found     = ovm.specifics.find(id);
    if (ovm_found == ovm.specifics.end()) {
      return false;
    }
    if (!ov.validReplacement(ovm_found->second)) {
      return false;
    }
  }
  return true;
}

} // namespace popart
