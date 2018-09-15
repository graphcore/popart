#ifndef NEURALNET_TENSOR_HPP
#define NEURALNET_TENSOR_HPP

#include <map>
#include <neuralnet/names.hpp>
#include <neuralnet/tensorinfo.hpp>

namespace neuralnet {

enum class TensorType {
  Activation = 0, // this includes Gradient tensors
  Const,
  //  Gradient,
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
  int n(Op *) const;
  // increment the number of times an Op consumes
  void increment(Op *);
  // decrement the number of times an Op consumes
  void decrement(Op *);
  // increment the current counts with those in this map
  void extend(const std::map<Op *, int> &);
  // return the total number of consumptions, taking
  // into account Ops which consume multiple times
  int getTotal() const;
  // the number of times each consumer uses the Tensor
  const std::map<Op *, int> &getMap() const;
  // the pointers to the consumers, no duplication for
  // Ops which consume multiple times
  std::vector<Op *> getOps();

private:
  std::map<Op *, int> consumers_m;
};

class TensorTypeInfo {

public:
  TensorTypeInfo(TensorType, std::string);
  TensorType type() const;
  const std::string &type_s() const;

private:
  TensorType tensorType_;
  std::string tensor_type_;
};

const std::map<TensorType, TensorTypeInfo> &getTensorTypeInfoMap();
std::map<TensorType, TensorTypeInfo> initTensorTypeInfoMap();

class Tensor {
public:
  // note : producer (if there is one)
  // must be set after construction
  Tensor(TensorId n, TensorType t, Graph *g);
  TensorId id;
  Graph *pgraph;
  // Activation, Variable, etc:
  TensorType tensorType() const;
  const std::string &tensor_type() const;
  Consumers consumers;
  // shape and data type. Not to be used be inferShape of pgraph has run
  TensorInfo info;

  Op *getProducer();
  void setProducer(Op *);
  bool hasProducer() const;

private:
  Op *producer;
  const TensorTypeInfo *tensorTypeInfo;
};
} // namespace neuralnet

#endif
