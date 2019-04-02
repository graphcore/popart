#ifndef GUARD_NEURALNET_SUBGRAPH_MATCH_HPP
#define GUARD_NEURALNET_SUBGRAPH_MATCH_HPP

#include "subgraphnames.hpp"
#include <iostream>
#include <vector>

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

  // bool valCompari(const Match &b) const {
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

  void setValue(float v) { value = v; }

  float getValue() const { return value; }

  // the indices in a schedule at which the sequences start at
  std::vector<Start> starts;

  // the length of the identical sequences
  int length;

private:
  float value = -1;
};

// true iff  v1 is a subset of v0
bool firstContainsSecond(const std::vector<int> &v0,
                         const std::vector<int> &v1);

std::ostream &operator<<(std::ostream &stream, const Match &match);

} // namespace subgraph
} // namespace fwtools

#endif
