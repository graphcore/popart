// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <sstream>
#include <string>
#include <popart/vertex.hpp>

namespace popart {

std::ostream &operator<<(std::ostream &ost, const PathToLoss &toLoss) {
  ost << "toLoss=";
  switch (toLoss) {
  case PathToLoss::Yes: {
    ost << 'Y';
    break;
  }
  case PathToLoss::No: {
    ost << 'N';
    break;
  }
  case PathToLoss::Undefined: {
    ost << 'U';
    break;
  }
  }
  return ost;
}

std::ostream &operator<<(std::ostream &ost, const PathFromLoss &fromLoss) {
  ost << "fromLoss=";
  switch (fromLoss) {
  case PathFromLoss::Yes: {
    ost << 'Y';
    break;
  }
  case PathFromLoss::No: {
    ost << 'N';
    break;
  }
  case PathFromLoss::Undefined: {
    ost << 'U';
    break;
  }
  }
  return ost;
}

std::ostream &operator<<(std::ostream &ost,
                         const ScheduledPreLoss &scheduledPreLoss) {
  ost << "schedPreLoss=";
  switch (scheduledPreLoss) {
  case ScheduledPreLoss::Yes: {
    ost << 'Y';
    break;
  }
  case ScheduledPreLoss::No: {
    ost << 'N';
    break;
  }
  case ScheduledPreLoss::Undefined: {
    ost << 'U';
    break;
  }
  }
  return ost;
}

std::string Vertex::wrtLossStr() const {
  std::ostringstream ss;
  ss << toLoss;
  ss << "  ";
  ss << fromLoss;
  ss << "  ";
  ss << scheduledPreLoss;
  return ss.str();
}

} // namespace popart
