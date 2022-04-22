// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <string>
#include <tuple>
#include <vector>
#include <popart/graphutils.hpp>
#include <popart/op/add.hpp>
#include <popart/op/conv.hpp>
#include <popart/op/dropoutbase.hpp>
#include <popart/op/groupnorm.hpp>
#include <popart/op/identity.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/reshape.hpp>
#include <popart/patterns/updateinplaceprioritiesforipu.hpp>
#include <popart/tensor.hpp>

#include "popart/error.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/patterns/patterns.hpp"

namespace popart {

namespace {

bool producedByMatMulOrConv(Tensor *t) {
  bool result = false;

  graphutils::traverse(
      {t},
      [&result](const Tensor *u) -> bool {
        // If producer is MatMul or Conv, stop.
        if (u->hasProducer()) {
          const auto producer = u->getProducerUnsafe();
          if (producer->isConvertibleTo<MatMulBaseOp>() ||
              producer->isConvertibleTo<ConvOp>()) {
            result = true;
            return false;
          }
        }

        return true;
      },
      [](const Op *op, const Tensor *u, const Tensor *v) -> bool {
        // Whitelist of ops, currently based on what bert requires.
        return op->isConvertibleTo<DropoutBaseOp>() ||
               op->isConvertibleTo<GroupNormOp>() ||
               op->isConvertibleTo<IdentityOp>() ||
               op->isConvertibleTo<ReshapeBaseOp>();
      },
      graphutils::TraversalType::DepthFirst,
      graphutils::VisitType::Pre,
      graphutils::TraversalDirection::Backward);

  return result;
}

void prioritiseBranchProducedByMatMulOrConv(AddOp &op) {
  for (auto &id_p : op.inplacePriorityDefault()) {
    OperatorIdentifier id = std::get<0>(id_p);
    float priority        = std::get<1>(id_p);

    InIndex argIndex;
    if (id == Onnx::CustomOperators::AddLhsInplace) {
      argIndex = op.getArg0InIndex();
    } else if (id == Onnx::CustomOperators::AddRhsInplace) {
      argIndex = op.getArg1InIndex();
    } else {
      throw error("unrecognised inplace op {}", id);
    }

    if (producedByMatMulOrConv(op.inTensor(argIndex))) {
      constexpr float arbitraryPositiveValue = 10.0f;
      priority += arbitraryPositiveValue;
      logging::pattern::trace("[UpdateInplacePrioritiesForIpu] For Op `{}`, "
                              "bumping priority of variant `{}`.",
                              op.debugName(),
                              id);
    }

    op.setInplacePriority(id, priority);
  }
}

} // namespace

void UpdateInplacePrioritiesForIpu::apply(Op *op) const {
  if (op->isConvertibleTo<AddOp>()) {
    logging::pattern::trace(
        "[UpdateInplacePrioritiesForIpu] Applying to Op `{}`.",
        op->debugName());
    prioritiseBranchProducedByMatMulOrConv(*dynamic_cast<AddOp *>(op));
  }
}

namespace {
static AddPatternName<UpdateInplacePrioritiesForIpu>
    registerName("UpdateInplacePrioritiesForIpu");
} // namespace

} // namespace popart
