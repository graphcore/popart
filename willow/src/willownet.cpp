#include <willow/graph.hpp>
#include <willow/willownet.hpp>

namespace willow {

WillowNet::WillowNet(std::string onnxModelFn,
                     const EarlyInfo &perk,
                     const DataFlow &df,
                     const std::vector<Loss *> &lossesIn,
                     const Optimizer *optimizerIn,
                     // Weights tensors which are not to be updated
                     const std::vector<TensorId> &cTens,
                     std::string logdir_,
                     const std::vector<std::string> &patternNames)
    : graph(new Graph({onnxModelFn,
                       perk,
                       df,
                       lossesIn,
                       optimizerIn,
                       cTens,
                       logdir_,
                       patternNames})) {}

void WillowNet::updateOptimizer(const Optimizer *optimizer) {
  graph->updateOptimizer(optimizer);
}

} // namespace willow
