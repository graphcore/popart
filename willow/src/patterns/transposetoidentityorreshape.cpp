// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <popart/graph.hpp>
#include <popart/graphutils.hpp>
#include <popart/op.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/reshape.hpp>
#include <popart/op/transpose.hpp>
#include <popart/patterns/transposetoidentityorreshape.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensorinfo.hpp>
#include <popart/topocons.hpp>

#include "popart/names.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/patterns/patterns.hpp"

namespace popart {

namespace {

std::vector<int64_t> unsqueezedDimensionsInReshape(Shape inShape,
                                                   Shape outShape) {
  std::vector<int64_t> dims;
  size_t in_iter = 0;
  for (size_t out_iter = 0; out_iter < outShape.size(); out_iter++) {
    if (inShape[in_iter] == outShape[out_iter]) {
      in_iter++;
    } else if (outShape[out_iter] == 1) {
      dims.push_back(out_iter);
    } else {
      return {};
    }
  }
  return dims;
}

bool dimensionsUnchangedInPermutation(std::vector<int64_t> dims, Shape perm) {
  for (auto dim : dims) {
    if (perm[dim] != dim) {
      return false;
    }
  }
  return true;
}

Shape removeDimensionsFromPermutation(std::vector<int64_t> dims, Shape perm) {
  // Assumption: The inputs have been validated with the function
  // "dimensionUnchangedInPermutation"
  Shape newPerm;
  for (auto dim : perm) {
    int64_t newDim = dim;
    bool remove    = false;
    for (auto removed_dim : dims) {
      if (dim == removed_dim) {
        remove = true;
        break;
      }
      if (dim > removed_dim) {
        newDim--;
      }
    }
    if (!remove) {
      newPerm.push_back(newDim);
    }
  }
  return newPerm;
}

std::pair<TransposeBaseOp *, std::vector<int64_t>>
findReverseTransposeWithReshapes(TransposeBaseOp *transpose) {
  TransposeBaseOp *reverseTranspose = nullptr;
  std::vector<int64_t> dimensionsToRemove;
  // Traverse Producers to find a transpose with the reverse permutation.
  graphutils::traverse(
      {transpose->inTensor(TransposeBaseOp::getInIndex())},
      [](Tensor *) -> bool { return true; },
      [&transpose, &reverseTranspose, &dimensionsToRemove](
          Op *candidate, Tensor *, Tensor *) -> bool {
        // Allow for Identity Ops between transposes.
        if (candidate->isConvertibleTo<IdentityOp>()) {
          if (candidate->input->n() > 1) {
            // Identity only has 1 input,
            // but this function has a hard assumption that it does.
            return false;
          }
          return true;
        }

        // Support 1 Reshape that adds singleton dimensions on the path between
        // transposes.
        // T35423: Support N reshapes. Support removing singleton dimensions.
        if (dimensionsToRemove.size() < 1 &&
            candidate->isConvertibleTo<ReshapeOp>()) {
          if (candidate->input->n() > 1) {
            // Reshape only has 1 input,
            // but this function has a hard assumption that it does.
            return false;
          }

          auto inShape = candidate->inInfo(ReshapeBaseOp::getInIndex()).shape();
          auto outShape =
              candidate->outInfo(ReshapeBaseOp::getOutIndex()).shape();
          if (inShape.size() > outShape.size()) {
            return false;
          }
          dimensionsToRemove = unsqueezedDimensionsInReshape(inShape, outShape);
          if (dimensionsToRemove.size() == 0) {
            return false;
          }
          // Check those dimensions are unaffected by permutation.
          return dimensionsUnchangedInPermutation(dimensionsToRemove,
                                                  transpose->getPerm());
        }

        auto candidateTranspose = dynamic_cast<TransposeBaseOp *>(candidate);
        if (candidateTranspose) {
          if (candidateTranspose->generateReversePermutation() ==
              removeDimensionsFromPermutation(dimensionsToRemove,
                                              transpose->getPerm())) {
            reverseTranspose = candidateTranspose;
          }
        }
        return false;
      },
      graphutils::TraversalType::DepthFirst,
      graphutils::VisitType::Pre,
      graphutils::TraversalDirection::Backward);
  return {reverseTranspose, dimensionsToRemove};
}

} // namespace

bool TransposeToIdentityOrReshapePattern::matches(Op *op) const {
  auto transpose = dynamic_cast<TransposeBaseOp *>(op);
  if (transpose && findReverseTransposeWithReshapes(transpose).first) {
    return true;
  }
  return false;
}

std::vector<const Tensor *>
TransposeToIdentityOrReshapePattern::touches(Op *) const {
  return {};
}

bool TransposeToIdentityOrReshapePattern::apply(Op *op) const {
  auto reverse_ignore =
      findReverseTransposeWithReshapes(dynamic_cast<TransposeBaseOp *>(op));
  auto reverse    = reverse_ignore.first;
  auto useReshape = reverse_ignore.second.size() > 0;

  auto in  = reverse->inTensor(TransposeBaseOp::getInIndex());
  auto out = op->outTensor(TransposeBaseOp::getOutIndex());

  OperatorIdentifier optype = useReshape ? Onnx::AiOnnx::OpSet9::Reshape
                                         : Onnx::AiOnnx::OpSet9::Identity;
  auto replacement = makeReplacementOpInIr(optype, op);

  op->disconnectAllInputs();
  op->disconnectAllOutputs();

  replacement->connectInTensor(0, in->id);
  replacement->connectOutTensor(0, out->id);
  if (useReshape) {
    dynamic_cast<ReshapeBaseOp *>(replacement)->setOutShape(out->info.shape());
  }
  replacement->setup();

  auto &graph = op->getGraph();
  graph.topoCons->transfer(op, replacement);

  graph.eraseOp(op->id);

  return true;
}

namespace {
static PatternCreator<TransposeToIdentityOrReshapePattern>
    PreUniReplPattern("TransposeToIdentityOrReshapePattern", true);
}

} // namespace popart
