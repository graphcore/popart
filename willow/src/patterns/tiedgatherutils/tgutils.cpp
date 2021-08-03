// Copyright(c) 2021 Graphcore Ltd.All rights reserved.
#include "tgutils.hpp"

#include <popart/graphutils.hpp>
#include <popart/op.hpp>
#include <popart/op/transpose.hpp>
#include <popart/tensor.hpp>

namespace popart {
namespace tgutil {

bool isProducedByTranspose(const Tensor *t) {
  return t->hasProducer() &&
         t->getProducer()->isConvertibleTo<TransposeBaseOp>();
}

// Finds the underlying variable by searching through producers.
Tensor *getVariable(Tensor *t) {
  Tensor *variable = nullptr;

  graphutils::traverse(
      {t},
      [&variable](Tensor *t) -> bool {
        if (t->tensorType() == TensorType::Variable ||
            t->tensorType() == TensorType::Const) {
          variable = t;
          return false;
        } else {
          return true;
        }
      },
      [](const Op *op, const Tensor *, const Tensor *) -> bool {
        return op->input->n() == 1;
      },
      graphutils::TraversalType::DepthFirst,
      graphutils::VisitType::Pre,
      graphutils::TraversalDirection::Backward);

  return variable;
}

} // namespace tgutil
} // namespace popart
