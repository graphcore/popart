#ifndef NEURALNET_TENSOR_HPP
#define NEURALNET_TENSOR_HPP

#include <map>
#include <willow/names.hpp>
#include <willow/tensorinfo.hpp>
#include <willow/vertex.hpp>

namespace willow {

enum class TensorType {
  ActGrad = 0, // an activation or a gradient, basically any output of an Op
  Const,
  Momentum,
  Stream,
  Unknown,
  Variable,
  N // number of tensor types
};

// The consumers (Ops) of a Tensor. Note that
// one Op may consume a Tensor at multiple locations.
class Consumers {
public:
  // Consumers is specific to a unique Tensor, which is stored
  // for later use in the constructor
  Consumers(Tensor *tensorConsumed_);
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
  std::vector<Op *> getOps() const;

  // if op is not in consumers_m : throw an error.
  // else, return a list of the other consumers which
  // MUST be inserted earlier than op in the topological
  // sort. This is DAG like, so if
  // a->b (b after a)
  // b->c (c after b)
  // then, consumersWhichTopoBefore(c)
  // can return {a,b} or just {b}.
  // This functionality was added to support in-place
  // ops and weight update ops
  std::vector<Op *> consumersWhichTopoBefore(Op *op) const;
  // There can be 1 op which MUST come after all others
  // This is a "strong" or "global" topo constraint
  void setTopoLast(Op *op);
  void removeTopoLast();
  bool hasTopoLast() const;
  Op *getTopoLast() const;
  // A weak topo constraint is a single edge in the
  // DAG, such as a->b in the above description.
  // It just states the relative order between 2 ops
  // Currently, we have no implementation of
  // weak topo cons (they'll be needed for in-place
  // though)
  bool hasWeakTopoCons() const;
  void append(std::stringstream &ss);

private:
  // The number of times an Op consumes the Tensor which
  // owns this Consumers
  std::map<Op *, int> consumers_m;
  Op *topoLast{nullptr};
  Tensor *tensorConsumed;
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
  Tensor(TensorId, TensorType, Ir *);
  TensorId id;
  Ir *pir;
  // ActGrad, Variable, etc:
  TensorType tensorType() const;
  const std::string &tensor_type() const;

  Consumers consumers;
  // shape and data type. Not to be used before inferShape of pir has run
  TensorInfo info;

  Op *getProducer();
  void setProducer(Op *);
  void resetProducer(Op *);
  bool hasProducer() const;

private:
  Op *producer;
  const TensorTypeInfo *tensorTypeInfo;
};

} // namespace willow

#endif
