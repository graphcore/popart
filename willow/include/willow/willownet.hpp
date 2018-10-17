#ifndef GUARD_NEURALNET_WILLOWNET_HPP
#define GUARD_NEURALNET_WILLOWNET_HPP

#include <willow/names.hpp>

namespace willow {

class WillowNet {
public:
  WillowNet(std::string fnOnnxModel,
            const EarlyInfo &,
            const DataFlow &,
            const std::vector<Loss *> &,
            const Optimizer *,
            const std::vector<std::string> &cTens,
            std::string logdir_,
            const std::vector<std::string> &patternNames);

  ~WillowNet();

  // update the optimizer. Note that the optimizer passed in
  // must be compatible with that in the constructor
  // Must call optimizerToDevice to take effect.
  void updateOptimizer(const Optimizer *);

  void setDevice(std::string x);

  Device & device();

private:
  // abstraction of the computation
  std::unique_ptr<Graph> graph {nullptr};
  std::unique_ptr<Device> device_ {nullptr};
};
} // namespace willow

#endif
