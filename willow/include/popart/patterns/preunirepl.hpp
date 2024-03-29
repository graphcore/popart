// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_PATTERNS_PREUNIREPL_HPP_
#define POPART_WILLOW_INCLUDE_POPART_PATTERNS_PREUNIREPL_HPP_

#include <vector>
#include <popart/patterns/pattern.hpp>

namespace popart {
class Op;
class Tensor;

// remove ()->[] where () is a Tensor and [] is an Op and ()->[]
// forms part of [.]->()->[]->(.). after this, this section will
// be [.]->(.). [] is the root op of the pattern.
// [] IS THE IDENTITY OP
// () must be consumed by only []. This constraint can be
//    removed, shouldn't be too hard, although care needed
//    with topo cons (as always)
class PreUniRepl : public PreAliasPattern {
public:
  PreUniRepl()           = default;
  ~PreUniRepl() override = default;

  // Pad with pad size zero
  // Sum with one input
  bool matches(Op *) const final;
  //  Only tensor (), which is deleted, is touched
  std::vector<const Tensor *> touches(Op *) const final;
  bool apply(Op *) const final;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_PATTERNS_PREUNIREPL_HPP_
