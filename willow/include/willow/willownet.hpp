#ifndef GUARD_NEURALNET_WILLOWNET_HPP
#define GUARD_NEURALNET_WILLOWNET_HPP

#include <map>
#include <willow/names.hpp>
#include <willow/tensorinfo.hpp>

namespace willow {

class Backend;
class Graph;
class DataFlow;
class EarlyInfo;
class Optimizer;
class Loss;

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

  // update the optimizer. Note that the optimizer passed in
  // must be compatible with that in the constructor
  // Must call optimizerToDevice to take effect.
  void updateOptimizer(const Optimizer *);

private:
  // abstraction of the computation
  std::unique_ptr<Graph> graph;
  // std::unique_ptr<Backend> backend;
};
} // namespace willow

#endif
