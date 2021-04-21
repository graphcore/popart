// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/bwdgraphinfo.hpp>

namespace popart {

bool ExpectedConnection::operator==(const ExpectedConnection &rhs) const {
  return fwdId == rhs.fwdId && type == rhs.type;
}

bool BwdGraphInfo::operator==(const BwdGraphInfo &rhs) const {
  return bwdGraphId == rhs.bwdGraphId && expectedInputs == rhs.expectedInputs &&
         expectedOutputs == rhs.expectedOutputs;
}

} // namespace popart
