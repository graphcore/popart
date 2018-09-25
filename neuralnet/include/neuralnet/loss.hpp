#ifndef GUARD_NEURALNET_LOSS_HPP
#define GUARD_NEURALNET_LOSS_HPP


#include <map>
#include <neuralnet/names.hpp>
#include <neuralnet/vertex.hpp>
#include <neuralnet/tensorinfo.hpp>

namespace neuralnet {
  enum class eLoss {NLL, L1};
  std::map<std::string, eLoss> initLossMap () ;
  const std::map<std::string, eLoss> & lossMap() ;
}





#endif
