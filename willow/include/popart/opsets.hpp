// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_OPSETS_HPP_
#define POPART_WILLOW_INCLUDE_POPART_OPSETS_HPP_

#include <vector>
#include <popart/names.hpp>

#include "popart/operatoridentifier.hpp"

namespace popart {

using OpsetVersion = unsigned;

// Returns the opid for a given opset domain and version.
// This is useful for finding which version of an op should be used depending on
// the opset version.
OperatorIdentifier
getOpid(const OpDomain &domain, OpsetVersion version, const OpType &opType);

std::vector<OperatorIdentifier> getOpset(int opsetVersion);

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_OPSETS_HPP_
