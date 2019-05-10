#include <map>
#include <sstream>

#include <poponnx/error.hpp>
#include <poponnx/vertex.hpp>

namespace poponnx {

std::map<Phase, std::string> init_phase_names() {
  return {{Phase::FWD, "FWD"},
          {Phase::BWD, "BWD"},
          {Phase::LOSS, "LOSS"},
          {Phase::UNDEFINED, "UNDEFINED"}};
}

const std::map<Phase, std::string> &phase_names() {
  const static std::map<Phase, std::string> M = init_phase_names();
  return M;
}

int Vertex::undefinedNPaths = -9;

void Vertex::incrNPathsToLoss() {
  if (nPathsToLoss_ == undefinedNPaths) {
    throw error(
        "Number of paths to loss of this Vertex not set in incrNPathsToLoss");
  }
  ++nPathsToLoss_;
}

void Vertex::setNPathsToLossToZero() { nPathsToLoss_ = 0; }

int Vertex::nPathsToLoss() const {
  if (nPathsToLoss_ == undefinedNPaths) {
    throw error("Number of paths to loss of this Vertex not set");
  }
  return nPathsToLoss_;
}

void Vertex::setPhase(Phase p) { phase_ = p; }

Phase Vertex::getPhase() const {
  //   if (phase_ == Phase::UNDEFINED) {
  //     throw error("This Vertex has an undefined phase");
  //   }
  return phase_;
}

void Vertex::setPathToBwd(PathToBwd p) { path_to_bwd_ = p; }

bool Vertex::hasPathToBwd() const {
  if (path_to_bwd_ == PathToBwd::UNDEFINED) {
    throw error("Vertex has undefined path to var update");
  } else if (path_to_bwd_ == PathToBwd::YES) {
    return true;
  } else {
    return false;
  }
}

bool Vertex::isFwdToBwd() const {
  return hasPathToBwd() && (getPhase() == Phase::FWD);
}

std::ostream &operator<<(std::ostream &ss, const Phase &phase) {
  switch (phase) {
  case Phase::FWD:
    ss << "Phase::FWD";
    break;
  case Phase::BWD:
    ss << "Phase::BWD";
    break;
  case Phase::LOSS:
    ss << "Phase::LOSS";
    break;
  case Phase::UNDEFINED:
    ss << "Phase::UNDEFINED";
    break;
  default:
    throw error("Unknown phase {}", static_cast<int>(phase));
  }

  return ss;
}

}; // namespace poponnx
