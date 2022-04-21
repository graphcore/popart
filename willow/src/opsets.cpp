// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include "opsets.gen.hpp"
#include <popart/operators.hpp>
#include <popart/opsets.hpp>

namespace popart {

OperatorIdentifier
getOpid(const OpDomain &argDomain, OpsetVersion version, const OpType &opType) {
  static OpsetMap opsets = getOpsets();

  OpDomain domain = argDomain;
  if (domain == "")
    domain = Domain::ai_onnx;

  auto foundOpset = opsets.find({domain, version});
  if (foundOpset != opsets.end()) {
    auto &opTypeMap = foundOpset->second;
    auto opFound    = opTypeMap.find(opType);
    if (opFound != opTypeMap.end()) {
      return opFound->second;
    } else {
      throw internal_error("Could not find an opid for op {} in opset {} {}",
                           opType,
                           domain,
                           version);
    }
  } else {
    std::stringstream ss;
    ss << "Domain, Version\n";
    for (auto &i : opsets) {
      ss << i.first.first << ", " << i.first.second << "\n";
    }

    throw internal_error(
        "Could not find domain and version {} {} in opsets. This could be due "
        "to the generated files being out of date. Current list of domains and "
        "versions is:\n{}",
        domain,
        version,
        ss.str());
  }
}

std::vector<OperatorIdentifier> getOpset(int opsetVersion) {
  static OpsetMap opsets = getOpsets();

  auto found = opsets.find({Domain::ai_onnx, opsetVersion});
  if (found != opsets.end()) {
    std::vector<OperatorIdentifier> opList;
    for (auto &opType_opId : found->second) {
      auto opType = opType_opId.first;
      auto opId   = opType_opId.second;

      opList.push_back(opId);
    }

    return opList;
  } else {
    throw error("Could not find opset {}", opsetVersion);
  }
}

} // namespace popart
