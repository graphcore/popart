#ifndef NEURALNET_TENSOR_HPP
#define NEURALNET_TENSOR_HPP

#include <map>
#include <neuralnet/names.hpp>
#include <neuralnet/tensorinfo.hpp>
#include <neuralnet/vertex.hpp>

namespace neuralnet {

enum class TensorType {
  ActGrad = 0, // an activation or a gradient, basically the output of an Op
  Const,
  //  Gradient,
  Momentum,
  //  Other,
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
  // into account Ops which consume multiple times,
  // so the sum over consuming nodes of the number of
  // times consumed
  int getTotal() const;
  // the number of times each consumer uses the Tensor
  const std::map<Op *, int> &getMap() const;
  // the pointers to the consumers, no duplication for
  // Ops which consume multiple times
  std::vector<Op *> getOps();

  // if op is not in consumers_m : throw an error.
  // else, return a list of the other consumers which
  // MUST be inserted earlier than op in the topological sort.
  // This functionality was added to support in-place ops
  std::vector<Op *> consumersWhichTopoBefore(Op *op);
  void setTopoFirst(Op *op);
  void removeTopoFirst();
  void setTopoLast(Op *op);
  void removeTopoLast();

private:
  // The number of times an Op consumes the Tensor which
  // owns this Consumers
  std::map<Op *, int> consumers_m;
  Op *topoFirst{nullptr};
  Op *topoLast{nullptr};
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

class Tensor : public Vertex {
public:
  // note : producer (if there is one)
  // must be set after construction
  Tensor(TensorId n, TensorType t, Graph *g);
  TensorId id;
  Graph *pgraph;
  // ActGrad, Variable, etc:
  TensorType tensorType() const;
  const std::string &tensor_type() const;
  Consumers consumers;
  // shape and data type. Not to be used be inferShape of pgraph has run
  TensorInfo info;

  Op *getProducer();
  void setProducer(Op *);
  void resetProducer(Op *);
  bool hasProducer() const;

private:
  Op *producer;
  const TensorTypeInfo *tensorTypeInfo;
};
} // namespace neuralnet

#endif
