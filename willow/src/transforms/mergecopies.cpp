// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <boost/range/algorithm.hpp>

#include <memory>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensors.hpp>
#include <popart/transforms/mergecopies.hpp>

namespace popart {

namespace {
// Get the Virtual Graph of Tensor t0 where t0 -> IpuCopyOp -> t1
int64_t getSrcTensorVirtualGraph(const Tensor *t1) {
  if (t1->hasProducer()) {
    auto producer = t1->getProducer();

    if (dynamic_cast<IpuCopyOp *>(producer)) {
      auto srcTensors = producer->input->tensors();

      if (srcTensors.size() == 1) {
        auto t0 = srcTensors[0];

        if (t0->hasVirtualGraphId()) {
          auto t0Id = t0->getVirtualGraphId();

          return t0Id;
        }
        throw internal_error(
            "Expected to be able to obtain VirtualGraphId of the "
            "source Tensor (t1 = {}) of an IpuCopyOp",
            t1->str());
      }
      throw internal_error(
          "Expected IpuCopyOp (producer = {}) in MergeCopies::apply to "
          "have exactly 1 input Tensor. Is this not the first time this "
          "transformation is being called?",
          producer->str());
    }
    throw internal_error("Expected the producer of a Tensor in a copy group to "
                         "be an IpuCopyOp, not an Op of another type ({})",
                         producer->str());
  }
  throw internal_error(
      "Expected a Tensor in a copy group to have a producer, in "
      "particular an IpuCopyOp producer. Tensor : {}",
      t1->str());
}
} // namespace

std::size_t MergeCopies::id() { return typeid(MergeCopies).hash_code(); }

static bool isCopyTensor(const Tensor *t) {
  return t->hasProducer() && t->getProducer()->isConvertibleTo<IpuCopyOp>();
}

static IpuCopyOp *createCopyOp(Graph &graph, uint64_t to_ipu) {
  Op::Settings settings(graph, "");
  auto ipuCopy_op = std::make_unique<IpuCopyOp>(
      Onnx::CustomOperators::IpuCopy, to_ipu, settings);
  auto ipuCopy = ipuCopy_op.get();
  graph.moveIntoGraph(std::move(ipuCopy_op));
  return ipuCopy;
}

static uint64_t getDestIpu(const std::vector<Tensor *> &copy_group) {
  auto p = copy_group.front()->getProducer();
  return dynamic_cast<IpuCopyOp *>(p)->getDestIpu();
}

static void mergeCopies(const std::vector<Tensor *> &copy_group, Graph &graph) {
  auto getSourceTensor = [](Tensor *t) {
    // assumes producer of t has 1 input only.
    return t->getProducer()->input->tensor(0);
  };

  // create a new copy op
  auto dest_ipu = getDestIpu(copy_group);
  auto copy_op  = createCopyOp(graph, dest_ipu);

  // move the copies
  for (auto t : copy_group) {
    auto source    = getSourceTensor(t);
    auto producer  = dynamic_cast<IpuCopyOp *>(t->getProducer());
    auto sourceIpu = producer->getSourceIpu(source->id);

    if (producer->input->n() != 1) {
      throw internal_error(
          "Attempting to merge a copy with more than one input!");
    }

    producer->disconnectInTensor(0, source);
    producer->disconnectOutTensor(t);

    int idx = copy_op->output->n();
    copy_op->connectInTensor(idx, source->id, sourceIpu);
    copy_op->connectOutTensor(idx, t->id);

    graph.eraseOp(producer->id);
  }

  copy_op->setup();
}

static std::vector<Op *> getOpsThatConsumeMultipleCopies(Graph &graph) {
  std::vector<Op *> ops;

  for (auto &id_op : graph.getOps()) {
    auto op = id_op.second.get();
    if (boost::count_if(op->input->tensors(), isCopyTensor) > 1) {
      ops.push_back(op);
    }
  }

  return ops;
}

// check that the op at position `op_schedule_iter`
// is the first consumer of `tensor` to appear in `op_schedule`
template <typename T>
static bool checkOpIsFirstConsumer(const T &op_schedule_iter,
                                   Tensor *tensor,
                                   const std::vector<Op *> &op_schedule) {
  for (auto consumer : tensor->consumers.getOps()) {
    if (std::find(op_schedule.begin(), op_schedule_iter, consumer) !=
        op_schedule_iter) {
      return false;
    }
  }

  return true;
}

static std::vector<Tensor *>
createCopyGroup(Op *op, const std::vector<Op *> &op_schedule) {
  std::vector<Tensor *> group;
  const auto op_schedule_iter =
      std::find(op_schedule.begin(), op_schedule.end(), op);

  for (auto tensor : op->input->tensors()) {
    if (isCopyTensor(tensor)) {
      auto producer = tensor->getProducer();
      // clang-format off
      if (producer->input->n() == 1
          && producer->output->n() == 1
          && checkOpIsFirstConsumer(op_schedule_iter, tensor, op_schedule)) {
        // clang-format on
        group.push_back(tensor);
      }
    }
  }

  return group;
}

bool MergeCopies::apply(Graph &graph) const {

  const auto multiple_copy_consumers = getOpsThatConsumeMultipleCopies(graph);

  const auto op_schedule = graph.getOpSchedule({});

  for (auto op : multiple_copy_consumers) {
    const auto copy_group = createCopyGroup(op, op_schedule);
    if (copy_group.size() > 1) {

      // without pipelining, we merge copies with different sources
      if (!graph.getIr().getSessionOptions().enablePipelining) {
        mergeCopies(copy_group, graph);
      }

      // with pipelining, we don't merge copies with different sources
      else {
        bool allSameSource = true;
        auto virtualGraph0 = getSrcTensorVirtualGraph(*copy_group.cbegin());
        for (auto t : copy_group) {
          if (getSrcTensorVirtualGraph(t) != virtualGraph0) {
            allSameSource = false;
          }
        }
        if (allSameSource) {
          mergeCopies(copy_group, graph);
        }
      }
    }
  }
  return true;
}

namespace {
bool init = Transform::registerTransform(new MergeCopies);
}

} // namespace popart
