#ifndef NEURALNET_VERTEX_HPP
#define NEURALNET_VERTEX_HPP

namespace neuralnet {

class Vertex {

public:
  Vertex()          = default;
  virtual ~Vertex() = default;
  void incrNPathsToLoss();
  int nPathsToLoss() const;

private:
  int nPathsToLoss_{0};
};

} // namespace neuralnet

#endif
