#include <neuralnet/error.hpp>
#include <neuralnet/vertex.hpp>

#include <sstream>

namespace neuralnet {

void Vertex::incrNPathsToLoss() {
  ++nPathsToLoss_;
}

int Vertex::nPathsToLoss() const{
  if (nPathsToLoss_ == 0){
    throw error("ILE : number of paths to loss is unset or 0");
  }
  return nPathsToLoss_;
}

};
