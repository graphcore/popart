#include <boost/range/algorithm.hpp>

#include <memory>
#include <poponnx/error.hpp>
#include <poponnx/graph.hpp>
#include <poponnx/names.hpp>
#include <poponnx/op.hpp>
#include <poponnx/op/ipucopy.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensorindex.hpp>
#include <poponnx/tensors.hpp>

#include <poponnx/transforms/mergecopies.hpp>

namespace poponnx {

std::size_t MergeCopies::id() { return typeid(MergeCopies).hash_code(); }

static bool isCopyTensor(const Tensor *t) {
  return t->hasProducer() && t->getProducer()->isConvertibleTo<IpuCopyOp>();
}

static IpuCopyOp *
createCopyOp(Graph &graph, uint64_t from_ipu, uint64_t to_ipu) {
  Op::Settings settings(graph, "");
  auto ipuCopy_op = std::make_unique<IpuCopyOp>(
      Onnx::CustomOperators::IpuCopy, from_ipu, to_ipu, settings);
  auto ipuCopy = ipuCopy_op.get();
  graph.moveIntoGraph(std::move(ipuCopy_op));
  return ipuCopy;
}

static void mergeCopies(const std::vector<Tensor *> &copy_group, Graph &graph) {
  std::set<IpuCopyOp *> producers;
  for (auto t : copy_group) {
    auto p = dynamic_cast<IpuCopyOp *>(t->getProducer());
    producers.insert(p);
  }

  std::map<Tensor *, Tensor *> dest_source_map;
  for (auto t : copy_group) {
    auto source = t->getProducer()->input->tensor(0);
    dest_source_map.insert({t, source});
  }

  // create a new copy op
  auto source_ipu = (*producers.begin())->getSourceIpu();
  auto dest_ipu   = (*producers.begin())->getDestIpu();
  auto copy_op    = createCopyOp(graph, source_ipu, dest_ipu);

  // move the copies
  for (auto t : copy_group) {
    auto source   = dest_source_map.at(t);
    auto producer = t->getProducer();

    producer->disconnectInTensor(source);
    producer->disconnectOutTensor(t);

    int idx = copy_op->output->n();
    copy_op->connectInTensor(idx, source->id);
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
      mergeCopies(copy_group, graph);
    }
  }

  return true;
}

namespace {
bool init = Transform::registerTransform(new MergeCopies);
}

} // namespace poponnx
