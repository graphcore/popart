#ifndef NEURALNET_TENSOR_HPP
#define NEURALNET_TENSOR_HPP

#include <map>
#include <neuralnet/names.hpp>

namespace neuralnet {

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

// The consumers (Ops) of a Tensor. Note that
// one Op may consume a Tensor at multiple locations.
class Consumers {
public:
  // The number of times an Op consumes a Tensor,
  // returns a non-negative integer
  int n(const Op *) const;
  // increment the number of times an Op consumes
  void increment(const Op *);
  // decrement the number of times an Op consumes
  void decrement(const Op *);
  // increment the current counts with those in this map
  void extend(const std::map<const Op *, int> &);
  // return the total number of consumptions, taking
  // into account Ops which consume multiple times
  int getTotal() const;
  const std::map<const Op *, int> &getMap() const;

private:
  std::map<const Op *, int> consumers_m;
};

class Tensor {
public:
  // note : producer (if there is one)
  // must be set after construction
  Tensor(TensorId n, TensorType t, Graph *g);
  TensorId id;
  Graph *pgraph;
  const TensorType type;
  const std::string tensor_type;
  Consumers consumers;
  Op *producer;
  std::vector<int64_t> shape;
  void append(std::stringstream &ss);
};
} // namespace neuralnet

#endif
