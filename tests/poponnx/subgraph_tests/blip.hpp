#ifndef GUARD_NEURALNET_BLIP_HPP
#define GUARD_NEURALNET_BLIP_HPP

#include <vector>
#include <poponnx/subgraph/subgraph.hpp>

namespace blip {

// We introduce the "Blip" class,
// a lightweight class for
// testing sub-graph matching.
class Blip;

// the "type" of a blip, which would correspond to the type
// of a neural net operation (such ScaleOp with scale=0.4)
using Type = int64_t;
// how valuable is the Blip in a sub-graph? The analogy
// with a neural net on an IPU is that ConvOp is more valuable
// than ReluOp
using Value = float;

using InIndex  = fwtools::subgraph::InIndex;
using OutIndex = fwtools::subgraph::OutIndex;
// We define the input of a Blip. Think neural net tensor:
//                          producer      index     tensor's name
using Input  = std::tuple<const Blip *, OutIndex, std::string>;
using Inputs = std::map<InIndex, Input>;

// 4 functions are needed to use
// the core algorithm in substring.hpp
class Blip {
public:
  Blip(Type t_, Value v_, const Inputs &ins_) : t(t_), v(v_), ins(ins_) {}

  // 1)
  // The core algorithm in substring.hpp
  // requires the identity to be a string.
  // We return the Type cast to a std::string
  fwtools::subgraph::EquivId getEquivId() const { return std::to_string(t); }

  // 2) where Inputs is a tuple<T*, OutIndex, std::string>
  const Inputs &getInputs() const { return ins; }

  // 3)
  std::vector<InIndex> getInIndices() const {
    std::vector<InIndex> in_indices;
    for (auto &x : getInputs()) {
      in_indices.push_back(x.first);
    }
    return in_indices;
  }

  // 4)
  float getValue() const { return v; }

  Inputs ins;

private:
  Type t;
  Value v;
};

struct Edge {
  Edge(int s, int d, OutIndex o, InIndex i)
      : sourceId(s), destId(d), outIndex(o), inIndex(i) {}
  int sourceId;
  int destId;
  OutIndex outIndex;
  InIndex inIndex;
};

} // namespace blip

#endif
