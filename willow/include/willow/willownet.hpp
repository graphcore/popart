#ifndef GUARD_NEURALNET_WILLOWNET_HPP
#define GUARD_NEURALNET_WILLOWNET_HPP

#include <string>
#include <willow/loss.hpp>

namespace willow {

class Graph;

class WillowNet {
public:
  WillowNet(std::string logDir,
            const EarlyInfo &,
            const DataFlow &dataFlow,
            const std::vector<Loss *> &);

  ~WillowNet();
  // create Graph
  void makeGraph();
  // connect to a backend (here, poplar with get an IPU)
  void connect(std::string backend);
  // compile for backend (here, poplar will build the poplar::Graph)
  void compile();
  // initialise weights on the backend
  void initialise();
  // take recommended steps of training
  void step();
  void getAnchors();
  void getWeights();

private:
  std::unique_ptr<Graph> graph;
};

} // namespace willow

#endif
