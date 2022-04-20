// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <string>
#include <vector>
#include <popart/op/randomnormal.hpp>
#include <popart/op/randomuniform.hpp>
#include <popart/op/zeros.hpp>
#include <popart/patterns/likeopspattern.hpp>
#include <popart/patterns/patterns.hpp>

namespace popart {

namespace {

// Replace RandomNormalLikeOp with the equivalent RandomNormalOp
static PatternCreator<LikeOpsPattern<RandomNormalLikeOp>>
    randomNormalLikeOpPattern("RandomNormalLikeOpPattern",
                              /* enabled = */ true,
                              /* mandatory = */ true);

// Replace RandomUniformLikeOp with the equivalent RandomUniformOp
static PatternCreator<LikeOpsPattern<RandomUniformLikeOp>>
    randomUniformLikeOpPattern("RandomUniformLikeOpPattern",
                               /* enabled = */ true,
                               /* mandatory = */ true);

// Replace ZerosLikeOP with the equivalent ZerosOp
static PatternCreator<LikeOpsPattern<ZerosLikeOp>>
    zerosLikeOpPattern("ZerosLikeOpPattern",
                       /* enabled = */ true,
                       /* mandatory = */ true);

} // namespace

} // namespace popart
