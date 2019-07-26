#ifndef GUARD_NEURALNET_ISOMORPHIC_HPP
#define GUARD_NEURALNET_ISOMORPHIC_HPP

#include "subgraphnames.hpp"
#include <map>
#include <vector>

namespace fwtools {
namespace subgraph {

template <typename T> std::vector<InIndex> getSubgraphInIndices(T *t) {
  std::vector<InIndex> in_indices;
  for (auto &x : t->getSubgraphInputs()) {
    in_indices.push_back(x.first);
  }
  return in_indices;
}

template <typename T> std::vector<OutIndex> getSubgraphOutIndices(T *t) {
  std::vector<InIndex> out_indices;
  for (auto &x : t->getSubgraphOutputs()) {
    out_indices.push_back(x.first);
  }
  return out_indices;
}

// Each edge into the sequence (sub-graph) is described by
// 1) an offset "delta" from the beginning of the sequence
// 2) the index "inIndex" at which the edge enters
struct InternalConsumer {
  InternalConsumer(int d_, InIndex i_) : delta(d_), inIndex(i_) {}
  int delta;
  InIndex inIndex;

  bool operator==(const InternalConsumer &other) const {
    return (delta == other.delta && inIndex == other.inIndex);
  }

  bool operator!=(const InternalConsumer &other) const {
    return !(*this == other);
  }
};

template <typename T>
int isomorphicUntil(int seq_length,
                    Start s0,
                    Start s1,
                    const std::vector<T *> &schedule,
                    const std::map<T *, int> &schedule_index) {

  auto relativeToStart = [&schedule_index](T *node, Start start) {
    Start index = schedule_index.at(node);
    return index - start;
  };

  // identifying a connection (a tensor for neural nets)
  // popart : (Op * creator, output index of creator, TensorId)
  // tf : similar, but the string can be empty
  using Input = std::tuple<T *, OutIndex, std::string>;

  // the producers of the inputs for a node in a sub-graph
  // are either in the sub-graph or not. When
  // 1) they are in the sub-graph, the corresponding producers
  //    in sub-graph 0 and sub-graph 1 must be at the same index
  //    relative to the start of the sub-graph, and the output
  //    index of the producers must be the same
  // 2) they aren't in the sub-graph, the corresponding consumers
  //    for the 2 sub-graphs must have identical consumers, at identical
  //    consumer InIndexs, for identical OutIndexs
  // also, the 2 sub-graphs must have an identical pattern of which
  // output indices are consumed externally (we might want to relax this)

  // case 2 : external producers. These maps should be identical
  std::map<Input, std::vector<InternalConsumer>> externProds0;
  std::map<Input, std::vector<InternalConsumer>> externProds1;

  for (int delta = 0; delta < seq_length; ++delta) {
    auto &t0 = schedule.at(s0 + delta);
    auto &t1 = schedule.at(s1 + delta);
    if ((getSubgraphInIndices(t0) != getSubgraphInIndices(t1)) ||
        getSubgraphOutIndices(t0) != getSubgraphOutIndices(t1)) {
      // this should actually be an error, as they shouldn't have
      // been returned as equivalent
      return delta;
    }

    auto &&ins0 = t0->getSubgraphInputs();
    auto &&ins1 = t1->getSubgraphInputs();

    for (auto inIndex : getSubgraphInIndices(t0)) {
      auto &in0 = ins0.at(inIndex);
      auto &in1 = ins1.at(inIndex);

      auto prod0 = std::get<0>(in0);
      auto out0  = std::get<1>(in0);

      auto prod1 = std::get<0>(in1);
      auto out1  = std::get<1>(in1);

      bool extern0 = (prod0 == nullptr || relativeToStart(prod0, s0) < 0);
      bool extern1 = (prod1 == nullptr || relativeToStart(prod1, s1) < 0);

      if (extern0 && extern1) {
        // both are external
        // 0
        auto found0 = externProds0.find(in0);
        if (found0 == externProds0.end()) {
          externProds0.insert({in0, {{delta, inIndex}}});
        } else {
          externProds0[in0].push_back({delta, inIndex});
        }
        // 1
        auto found1 = externProds1.find(in1);
        if (found1 == externProds1.end()) {
          externProds1.insert({in1, {{delta, inIndex}}});
        } else {
          externProds1[in1].push_back({delta, inIndex});
        }

        // we check if they are still isomorphic
        // with this external input now included
        if (externProds0.at(in0) != externProds1.at(in1)) {
          return delta;
        }
      }

      // both are internal
      else if (!extern0 && !extern1) {
        auto rel0 = relativeToStart(prod0, s0);
        auto rel1 = relativeToStart(prod1, s1);
        if (rel0 != rel1 || out0 != out1) {
          return delta;
        }
      }

      // one is internal, one is external
      else {
        return delta;
      }
    }

    // we check that the output indices which are consumed are identical
    for (auto outIndex : getSubgraphOutIndices(t0)) {
      bool externOut0         = false;
      auto &&subgraphOutputs0 = t0->getSubgraphOutputs();
      auto consumers0         = subgraphOutputs0.at(outIndex);
      for (auto con0 : consumers0) {
        // if con0 is not in schedule_index, then it is not
        // in the schedule, which means it is definitely external to
        // this subgraph (as this subgraph is contiguous in the schedule)
        if (schedule_index.find(con0) == schedule_index.end() ||
            relativeToStart(con0, s0) >= seq_length) {
          externOut0 = true;
        }
      }

      bool externOut1         = false;
      auto &&subgraphOutputs1 = t1->getSubgraphOutputs();
      auto consumers1         = subgraphOutputs1.at(outIndex);
      for (auto con1 : consumers1) {
        if (schedule_index.find(con1) == schedule_index.end() ||
            relativeToStart(con1, s1) >= seq_length) {
          externOut1 = true;
        }
      }
      if (externOut0 != externOut1) {
        return delta;
      }
    }
  }

  return seq_length;
}

template <typename T>
bool areIsomorphic(int seq_length,
                   Start s0,
                   Start s1,
                   const std::vector<T *> &schedule,
                   const std::map<T *, int> &schedule_index) {

  return (isomorphicUntil(seq_length, s0, s1, schedule, schedule_index) ==
          seq_length);
}

} // namespace subgraph
} // namespace fwtools

#endif
