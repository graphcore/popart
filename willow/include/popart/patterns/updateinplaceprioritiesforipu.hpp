// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_PATTERNS_UPDATEINPLACEPRIORITIESFORIPU_HPP_
#define POPART_WILLOW_INCLUDE_POPART_PATTERNS_UPDATEINPLACEPRIORITIESFORIPU_HPP_

#include <popart/patterns/pattern.hpp>

namespace popart {
class Op;

/**
 * \brief For AddOps, prioritise the inplace variant for the branch(es) that
 * have a Conv or MatMul in their producers.
 *
 * \details Note, this means if both branches have a Conv or MatMul, both will
 * have their priority bumped, resulting in a net zero effect. Finally, whilst
 * traversing back through the producers, only certain ops are traversable. See
 * the implementation of #apply for the exact details.
 */
class UpdateInplacePrioritiesForIpu : public Pattern {
public:
  void apply(Op *) const;
};

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_PATTERNS_UPDATEINPLACEPRIORITIESFORIPU_HPP_
