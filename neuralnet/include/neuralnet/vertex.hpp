#ifndef NEURALNET_VERTEX_HPP
#define NEURALNET_VERTEX_HPP

namespace neuralnet {


class Vertex {

public:
  Vertex()          = default;
  virtual ~Vertex() = default;
  Vertex(const Vertex &) : nPathsToLoss_(-100) {}
  Vertex &operator=(const Vertex &) = delete;


  void incrNPathsToLoss();
  int nPathsToLoss() const;
  void setNPathsToLossToZero() {
    nPathsToLoss_ = 0;
  }

private:
  int nPathsToLoss_{-100};
};

} // namespace neuralnet

#endif
