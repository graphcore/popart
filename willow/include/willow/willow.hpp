#ifndef GUARD_NEURALNET_NEURALNET_HPP
#define GUARD_NEURALNET_NEURALNET_HPP

#include <string>
#include <willow/loss.hpp>

namespace willow {

class Graph;

class Willow {
public:
  // TODO: include with anchors the frequency
  //       at which they are to be read.
  // TODO: there will be some nice async
  //       thread programming for this class.
  Willow(std::string logDir, const std::vector<Loss *> &);
  ~Willow();
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
