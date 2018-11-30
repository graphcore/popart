#include <poponnx/error.hpp>
#include <poponnx/intervals.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/names.hpp>
#include <poponnx/op.hpp>
#include <poponnx/pbwrap.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensornames.hpp>

#include <poponnx/transforms/recompute.hpp>

namespace poponnx {

namespace {

Op *growRecomputeOp(Ir &ir, Op *oriOp, const std::set<Op *> &checkpoints) {

  // the recompute op:
  OpId rcId = ir.moveIntoIr(oriOp->clone());

  Op *rcOp = ir.getOps().at(rcId).get();

  // set inputs and outputs of  the new Op.
  std::map<int, TensorId> inputs;
  for (auto &index_tensor : oriOp->input->tensorMap()) {
    int index      = index_tensor.first;
    Tensor *tensor = index_tensor.second;
    // if the tensor was produced by a non-checkpointed op,
    // we need to use the recomputed version of it
    if (tensor->hasProducer() &&
        checkpoints.count(tensor->getProducer()) == 0) {
      inputs[index] = getRecompId(tensor->id);
    } else {
      inputs[index] = tensor->id;
    }
  }
  ir.connectInputsFromInputMapWrapper(InputMapWrapper(inputs), rcId);

  std::map<int, TensorId> outputs;
  for (auto &index_tensor : oriOp->output->tensorMap()) {
    int index            = index_tensor.first;
    const Tensor *tensor = index_tensor.second;
    outputs[index]       = getRecompId(tensor->id);
  }
  ir.connectOutputsFromOutputMapWrapper(OutputMapWrapper(outputs), rcId);
  rcOp->setup();

  // yank down the priority of the new Op
  // (must be run as late as possible):
  rcOp->priority = std::numeric_limits<double>::lowest();

  // oriOp's outputs should not be consumed by grad op:
  for (auto &ind_ten : oriOp->output->tensorMap()) {
    Tensor *oriTen = ind_ten.second;
    Tensor *recTen = ir.getTensors().get(getRecompId(oriTen->id));
    for (auto &con : oriTen->consumers.getOps()) {
      if (con->getPhase() == Phase::BWD) {
        for (auto &con_ind_ten : con->input->tensorMap()) {
          int gradIn = con_ind_ten.first;
          if (con_ind_ten.second == oriTen) {
            con->input->reset(gradIn, recTen);
            recTen->consumers.increment(con);
            oriTen->consumers.decrement(con);
          }
        }
      }
    }
  }

  // note: oriOp will still be pointed to
  // by grad op as it's creator. This design
  // choice might need revision.

  return rcOp;
}

} // namespace

std::size_t Recompute::id() { return typeid(Recompute).hash_code(); }

bool Recompute::apply(Ir &ir) const {
  std::vector<Op *> fwdOps;
  for (auto op : ir.getOpSchedule({})) {
    if (op->isFwdToBwd()) {
      fwdOps.push_back(op);
    }
  }

  // liveSets[i] : set of ops whose outputs have not all
  // been consumed by their (non-grad) consumers just after
  // linearised[i] has run. By this defn,
  // linearised[i] \in live[i]
  std::vector<std::set<Op *>> liveSets = ir.getLiveSets(fwdOps);

  // The memory (bytes) which will be needed to
  // store all the output tensors in a liveness set.
  std::vector<int64_t> memoryOfLives;
  for (auto &liveSet : liveSets) {
    int64_t mem = 0;
    for (auto op : liveSet) {
      mem += op->memOfOutputs();
    }
    memoryOfLives.push_back(mem);
  }

  int nFwdOps = static_cast<int>(fwdOps.size());
  if (nFwdOps != liveSets.size() || memoryOfLives.size() != nFwdOps) {
    throw error("ILE : sizes of vectors do not match");
  }

  // TODO (see T5099)
  // this should change. resnet-50 has way more memory for early layers.
  // see
  // https://github.com/albanie/convnet-burden/blob/master/reports/resnet18.md
  // It should take in memoryOfLives, make intervals on cumulative memory.
  std::vector<std::array<int, 2>> intervals = getDecreasingIntervals(nFwdOps);

  //   defn, checkpoints: Ops whose
  //   outputs we guarantee will be available
  //   at any time
  std::set<Op *> checkpoints;

  // we choose the lowest memory set from each interval,
  // and add its members to checkpoints.
  for (auto interval : intervals) {
    int begin            = interval[0];
    int end              = interval[1];
    int64_t lowestMemory = std::numeric_limits<int64_t>::max();
    std::set<Op *> bestSet{};
    for (int i = begin; i < end; ++i) {
      if (memoryOfLives[i] < lowestMemory) {
        lowestMemory = memoryOfLives[i];
        bestSet      = liveSets[i];
      }
    }
    for (Op *op : bestSet) {
      if (checkpoints.count(op) == 0) {
        checkpoints.insert(op);
      }
    }
  }

  // all non-checkpoint pre-loss nodes.
  std::vector<Op *> nonCheckpoints;
  for (auto &op : fwdOps) {
    if (checkpoints.count(op) == 0) {
      nonCheckpoints.push_back(op);
    }
  }

  for (auto &op : nonCheckpoints) {
    growRecomputeOp(ir, op, checkpoints);
  }

  return true;
}

namespace {
bool init = Transform::registerTransform(new Recompute);
}

} // namespace poponnx
