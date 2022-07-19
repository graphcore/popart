// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef POPART_TESTS_INTEGRATION_SUBGRAPH_TESTS_BLIP_HPP_
#define POPART_TESTS_INTEGRATION_SUBGRAPH_TESTS_BLIP_HPP_

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include "popart/subgraph/subgraphnames.hpp"

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
//                        producer      index     tensor's name
using Input  = std::tuple<const Blip *, OutIndex, std::string>;
using Inputs = std::map<InIndex, Input>;

// given all Inputs, we can infer all Outputs
//                         consumers (unique)         output index
using Outputs = std::map<OutIndex, std::set<const Blip *>>;

// 4 functions are needed to use
// the core algorithm in substring.hpp
class Blip {
public:
  Blip(Type t_, Value v_, const Inputs &ins_) : t(t_), v(v_), ins(ins_) {}

  // 1)
  // The core algorithm in substring.hpp
  // requires the identity to be a string.
  // We return the Type cast to a std::string
  fwtools::subgraph::EquivId getSubgraphEquivId() const {
    return std::to_string(t);
  }

  // 2) where Inputs is a tuple<T*, OutIndex, std::string>
  const Inputs &getSubgraphInputs() const { return ins; }

  // 3)
  float getSubgraphValue() const { return v; }

  // 4)
  const Outputs &getSubgraphOutputs() const { return outs; }

  void addIn(InIndex in_index, const Blip *source, OutIndex out_index) {
    ins[in_index] = {source, out_index, ""};
  }

  void addOut(const Blip *dest, OutIndex out_index) {
    if (outs.find(out_index) == outs.end()) {
      outs[out_index] = {};
    }
    outs[out_index].insert(dest);
  }

private:
  Outputs outs;
  Type t;
  Value v;
  Inputs ins;
};

struct Edge {
  Edge(int s, int d, OutIndex o, InIndex i)
      : sourceId(s), destId(d), outIndex(o), inIndex(i) {}
  int sourceId;
  int destId;
  OutIndex outIndex;
  InIndex inIndex;
};

class ModThreeCostModel {
public:
  float value(int64_t begin,
              int64_t end,
              const std::vector<const Blip *> &sched,
              const std::map<const Blip *, int> &sched_index) {
    // prefer lengths of 2 (modulo 3)
    // tie-break   1 : prefer long ones
    // tie-breaker 2 : prefer late begins
    //
    float val = (end - begin) % 3 + (end - begin) * 1e-3 + 1e-6 * begin;
    return val;
  }
};

} // namespace blip

#endif // POPART_TESTS_INTEGRATION_SUBGRAPH_TESTS_BLIP_HPP_
