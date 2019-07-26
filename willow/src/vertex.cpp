#include <map>
#include <sstream>

#include <popart/error.hpp>
#include <popart/vertex.hpp>

namespace popart {

std::string Vertex::wrtLossStr() const {

  std::ostringstream ss;

  ss << "toLoss=";
  switch (toLoss) {
  case PathToLoss::Yes: {
    ss << 'Y';
    break;
  }
  case PathToLoss::No: {
    ss << 'N';
    break;
  }
  case PathToLoss::Undefined: {
    ss << 'U';
    break;
  }
  }

  ss << "  fromLoss=";
  switch (fromLoss) {
  case PathFromLoss::Yes: {
    ss << 'Y';
    break;
  }
  case PathFromLoss::No: {
    ss << 'N';
    break;
  }
  case PathFromLoss::Undefined: {
    ss << 'U';
    break;
  }
  }

  ss << "  schedPreLoss=";
  switch (scheduledPreLoss) {
  case ScheduledPreLoss::Yes: {
    ss << 'Y';
    break;
  }
  case ScheduledPreLoss::No: {
    ss << 'N';
    break;
  }
  case ScheduledPreLoss::Undefined: {
    ss << 'U';
    break;
  }
  }

  return ss.str();
}

}; // namespace popart
