// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_PREUNIREPL_HPP
#define GUARD_NEURALNET_PREUNIREPL_HPP

#include <popart/patterns/pattern.hpp>

namespace popart {

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

#endif
