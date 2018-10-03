#include <neuralnet/error.hpp>
#include <neuralnet/vertex.hpp>

#include <sstream>

namespace neuralnet {

void Vertex::incrNPathsToLoss() {
  if (nPathsToLoss_ < 0) {
    nPathsToLoss_ = 1;
  } else {
    ++nPathsToLoss_;
  }
}

int Vertex::nPathsToLoss() const { return nPathsToLoss_; }

}; // namespace neuralnet
