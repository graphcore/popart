#include <willow/error.hpp>
#include <willow/vertex.hpp>

#include <sstream>

namespace willow {

void Vertex::incrNPathsToLoss() {
  if (nPathsToLoss_ < 0) {
    nPathsToLoss_ = 1;
  } else {
    ++nPathsToLoss_;
  }
}

int Vertex::nPathsToLoss() const { return nPathsToLoss_; }

}; // namespace willow
