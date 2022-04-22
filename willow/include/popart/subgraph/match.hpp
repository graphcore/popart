// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SUBGRAPH_MATCH_HPP
#define GUARD_NEURALNET_SUBGRAPH_MATCH_HPP

#include <iosfwd>
#include <vector>
#include <popart/subgraph/subgraphnames.hpp>

namespace fwtools {
namespace subgraph {

class Match {
public:
  Match(const std::vector<Start> &s, int l);

  // the length of the sequence (sub-graph) takes
  // highest priority in the comparison, after values
  bool setComparison(const Match &b) const {
    if (length < b.length || length > b.length) {
      return length < b.length;
    }
    if (starts.size() != b.starts.size()) {
      return starts.size() < b.starts.size();
    }
    return starts < b.starts;
  }

  bool operator<(const Match &b) const {
    if (getValue() < b.getValue() || getValue() > b.getValue()) {
      return getValue() < b.getValue();
    }
    return setComparison(b);
  }

  bool operator==(const Match &other) const {
    return length == other.length && starts == other.starts;
  }

  // an element of other is also in this
  bool intersects(const Match &other) const;

  // no element of other is not in this Match
  bool contains(const Match &other) const;

  // this Match's starts are a superset of startsRhs
  bool containsStarts(const std::vector<Start> &startsRhs) const;

  // this Match's starts intersect with startsRhs
  bool startsIntersect(const std::vector<Start> &startsRhs) const;

  // contains other and same size (number of starts) as other
  bool subsumes(const Match &other) const;

  //  xxxx         xxxx     xx         xx
  //     xxxxx      xxxx   xx      xxxxx
  bool crosses(const Match &other) const;

  // rhs maps to this in a repeated way.
  bool fitsCleanly(const Match &rhs) const;

  void setValue(double v) { value = v; }

  double getValue() const { return value; }

  void setDiscountedValue(double v) { discountedValue = v; }

  double getDiscountedValue() const { return discountedValue; }

  // the indices in a schedule at which the sequences start at
  std::vector<Start> starts;

  // the length of the identical sequences
  int length;

private:
  // Value: Strictly positive value of this match, representing the code size
  // reduction benefit of outlining a sequence of operators
  // For two matches:
  // A: xxxxx
  // B:  xxx
  // where one is subsumed by the other, it must hold that value(A) >= value(B)
  double value = -1;
  // Discounted value: Value of this match after accounting for other factors
  // than code size reduction, such as software parallelism.
  // This value can be negative, and a subsumed child match (B) can have a
  // higher value than it's parent match (A).
  double discountedValue = -1;
};

// true iff  v1 is a subset of v0
bool firstContainsSecond(const std::vector<int> &v0,
                         const std::vector<int> &v1);

std::ostream &operator<<(std::ostream &stream, const Match &match);

} // namespace subgraph
} // namespace fwtools

#endif
