// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <map>
#include <ostream>
#include <string>
#include <vector>
#include <popart/bwdgraphinfo.hpp>
#include <popart/error.hpp>
#include <popart/logging.hpp>

#include "popart/graphid.hpp"
#include "popart/names.hpp"

namespace popart {

bool ExpectedConnection::operator==(const ExpectedConnection &rhs) const {
  return fwdId == rhs.fwdId && type == rhs.type;
}

bool BwdGraphInfo::operator==(const BwdGraphInfo &rhs) const {
  return bwdGraphId == rhs.bwdGraphId && expectedInputs == rhs.expectedInputs &&
         expectedOutputs == rhs.expectedOutputs;
}

std::ostream &operator<<(std::ostream &out, ExpectedConnectionType type) {
  switch (type) {
  case ExpectedConnectionType::Fwd: {
    out << "Fwd";
    break;
  }
  case ExpectedConnectionType::FwdGrad: {
    out << "FwdGrad";
    break;
  }
  default: {
    throw error("Unsupported ExpectedConnectionType value");
  }
  }

  return out;
}

std::ostream &operator<<(std::ostream &out, const ExpectedConnection &ec) {
  out << "{" << ec.fwdId << ", " << ec.type << "}";
  return out;
}

std::ostream &operator<<(std::ostream &out, const ExpectedConnections &ecs) {
  out << "{" << logging::join(ecs.begin(), ecs.end(), ", ") << "}";
  return out;
}

std::ostream &operator<<(std::ostream &out, const BwdGraphInfo &bgi) {
  out << "{"
      << "bwdGraphId: " << bgi.bwdGraphId << ", "
      << "expectedInputs: " << bgi.expectedInputs << ", "
      << "expectedOutputs: " << bgi.expectedOutputs << "}";
  return out;
}

std::ostream &operator<<(std::ostream &out,
                         const FwdGraphToBwdGraphInfo::value_type &fgtbgi) {
  out << fgtbgi.first << ": " << fgtbgi.second;
  return out;
}

std::ostream &operator<<(std::ostream &out,
                         const FwdGraphToBwdGraphInfo &fgtbgi) {
  out << "{" << logging::join(fgtbgi.begin(), fgtbgi.end(), ", ") << "}";
  return out;
}

} // namespace popart
