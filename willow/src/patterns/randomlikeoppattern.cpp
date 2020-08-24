// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <popart/op/randomnormal.hpp>
#include <popart/op/randomuniform.hpp>
#include <popart/patterns/patterns.hpp>
#include <popart/patterns/randomlikeoppattern.hpp>

namespace popart {

namespace {

// Replace RandomNormalLikeOp with the equivalent RandomNormalOp
static PatternCreator<RandomLikeOpPattern<RandomNormalLikeOp>>
    randomNormalLikeOpPattern(PreAliasPatternType::RandomNormalLikeOpPattern,
                              "RandomNormalLikeOpPattern",
                              /* enabled = */ true,
                              /* mandatory = */ true);

// Replace RandomUniformLikeOp with the equivalent RandomUniformOp
static PatternCreator<RandomLikeOpPattern<RandomUniformLikeOp>>
    randomUniformLikeOpPattern(PreAliasPatternType::RandomUniformLikeOpPattern,
                               "RandomUniformLikeOpPattern",
                               /* enabled = */ true,
                               /* mandatory = */ true);

} // namespace

} // namespace popart