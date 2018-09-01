#ifndef NEURALNET_TENSOR_HPP
#define NEURALNET_TENSOR_HPP

#include <neuralnet/names.hpp>


namespace neuralnet{


enum class TensorType {
  Activation = 0,
  Const,
  Gradient,
  Momentum,
  Other,
  Stream,
  Unknown,
  Variable,
  N // number of tensor types
};

class Op;
class Graph;

class Tensor {
public:
  // note : cannot rely on producer to get pgraph, as producer might be nullptr
  Tensor(TensorId n, TensorType t, Graph *g)
      : name(n), pgraph(g), type(t),
        tensor_type(pgraph->tensorTypes.asString(t)), producer(nullptr) {
        }
  TensorId name;
  Graph *pgraph;
  const TensorType type;
  const std::string tensor_type;
  Op *producer;
  std::vector<Op *> consumers;
  std::vector<int64_t> shape;
  void append(std::stringstream &ss);
};
}

#endif
