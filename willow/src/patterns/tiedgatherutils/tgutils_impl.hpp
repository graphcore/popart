// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <popart/graph.hpp>
#include <popart/graphutils.hpp>
#include <popart/ir.hpp>
#include <popart/op/accumulate.hpp>
#include <popart/op/adamupdater.hpp>
#include <popart/op/adamvarupdate.hpp>
#include <popart/op/collectives/replicatedallgather.hpp>
#include <popart/op/collectives/replicatedreducescatter.hpp>
#include <popart/op/dropout.hpp>
#include <popart/op/matmul.hpp>
#include <popart/op/mul.hpp>
#include <popart/op/slice.hpp>
#include <popart/op/transpose.hpp>
#include <popart/opidentifier.hpp>
#include <popart/topocons.hpp>

namespace popart {
namespace tgutil {

template <typename T> T *weightConsumedBy(Tensor *w) {
  w = getVariable(w);
  if (w) {
    return searchConsumersFor<T>(w);
  }
  return nullptr;
}

template <class T, ExecutionContext Ctx> T *searchConsumersFor(Tensor *t) {
  T *result = nullptr;

  graphutils::traverse(
      {t},
      [&result](Tensor *t) -> bool {
        auto op = t->getProducerUnsafe();
        if (op && op->isConvertibleTo<T>() &&
            op->settings.executionContext == Ctx) {
          result = dynamic_cast<T *>(op);
          return false;
        } else {
          return true;
        }
      },
      [](const Op *op, const Tensor *t, const Tensor *u) -> bool {
        return (op->isConvertibleTo<DropoutGradOp>() &&
                op->outTensor(DropoutGradOp::getOutIndex())->id == u->id) ||
               (op->isConvertibleTo<ReplicatedReduceScatterOp>() &&
                op->outTensor(ReplicatedReduceScatterOp::getOutIndex())->id ==
                    u->id) ||
               // TODO(T42598): Improve this as it's too general. Most ops that
               // have one input and one output are view changing.
               (op->input->n() == 1 && op->output->n() == 1) ||
               (op->isConvertibleTo<T>() &&
                op->settings.executionContext == Ctx);
      },
      graphutils::TraversalType::BreadthFirst,
      graphutils::VisitType::Pre,
      graphutils::TraversalDirection::Forward);

  return result;
}

template <class T, ExecutionContext Ctx> T *searchProducersFor(Tensor *t) {
  T *result = nullptr;

  graphutils::traverse(
      {t},
      [](Tensor *) -> bool { return true; },
      [&result](Op *op, const Tensor *t, const Tensor *u) -> bool {
        if (op->isConvertibleTo<T>() && op->settings.executionContext == Ctx) {
          result = dynamic_cast<T *>(op);
          return false;
        }

        const auto &uId = u->id;
        return op->input->n() == 1 ||
               (op->isConvertibleTo<AdamUpdaterOp>() &&
                op->inId(AdamUpdaterOp::getAccl1InIndex()) == uId) ||
               (op->isConvertibleTo<AdamVarUpdateOp>() &&
                op->inId(AdamVarUpdateOp::getUpdaterInIndex()) == uId) ||
               (op->isConvertibleTo<AccumulateBaseOp>() &&
                op->inId(AccumulateBaseOp::getUpdaterInIndex()) == uId) ||
               (op->isConvertibleTo<DropoutGradOp>() &&
                op->inId(DropoutGradOp::getGradInIndex()) == uId) ||
               // Grad Unscaling for Adam-based optimizers
               (op->isConvertibleTo<MulOp>() &&
                op->inId(MulOp::getArg0InIndex()) == uId) ||
               // Replicated Tensor Sharding
               (op->isConvertibleTo<ReplicatedReduceScatterOp>() &&
                op->inId(ReplicatedReduceScatterOp::getInIndex()) == uId) ||
               // Replicated Tensor Sharding
               (op->isConvertibleTo<ReplicatedAllGatherOp>() &&
                op->inId(ReplicatedAllGatherOp::getInIndex()) == uId);
      },
      graphutils::TraversalType::DepthFirst,
      graphutils::VisitType::Pre,
      graphutils::TraversalDirection::Backward);

  return result;
}

template <class T, ExecutionContext Ctx>
std::vector<T *> findAllConsumers(Tensor *w) {
  std::vector<T *> result;

  graphutils::traverse(
      {w},
      [](Tensor *) -> bool { return true; },
      [&result](Op *op, const Tensor *t, const Tensor *u) -> bool {
        if (op->isConvertibleTo<T>() && op->settings.executionContext == Ctx) {
          result.push_back(static_cast<T *>(op));
        }

        return (op->isConvertibleTo<MatMulOp>() &&
                op->outId(MatMulOp::getOutIndex()) == u->id) ||
               (op->isConvertibleTo<ReplicatedReduceScatterOp>() &&
                op->outId(ReplicatedReduceScatterOp::getOutIndex()) == u->id) ||
               // Most ops that have one input and one output are view changing.
               (op->input->n() == 1 && op->output->n() == 1);
      },
      graphutils::TraversalType::BreadthFirst,
      graphutils::VisitType::Pre,
      graphutils::TraversalDirection::Forward);

  return result;
}

template <class T> Tensor *maybeTraverseProducer(InIndex index, Tensor *t) {
  if (t->hasProducer() && t->getProducer()->isConvertibleTo<T>()) {
    return t->getProducer()->inTensor(index);
  }
  return t;
}

} // namespace tgutil
} // namespace popart
