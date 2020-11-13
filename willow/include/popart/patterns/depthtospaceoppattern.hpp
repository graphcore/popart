// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_DEPTHTOSPACE_OP_PATTERN_HPP
#define GUARD_NEURALNET_DEPTHTOSPACE_OP_PATTERN_HPP

#include <popart/patterns/patterns.hpp>

namespace popart {

// Replace DepthToSpaceOp with
// reshape and transpose.
class DepthSpaceOpPattern : public PreAliasPattern {

protected:
  void transform(const Shape &shapeIn,
                 const std::vector<int64_t> &shape6D,
                 int64_t blocksize,
                 Tensor *input,
                 Tensor *output,
                 Op *depthSpace,
                 Graph &) const;

private:
  Op *transposeDepthSpace(const std::vector<int64_t> &perm,
                          Tensor *input,
                          Op *depthSpace,
                          Op *reshape6D,
                          Graph &) const;
};

class DepthToSpaceOpPattern : public DepthSpaceOpPattern {
public:
  bool matches(Op *) const override;
  std::vector<const Tensor *> touches(Op *) const override;
  bool apply(Op *) const override;
};

class SpaceToDepthOpPattern : public DepthSpaceOpPattern {
public:
  bool matches(Op *) const override;
  std::vector<const Tensor *> touches(Op *) const override;
  bool apply(Op *) const override;
};

} // namespace popart

#endif
