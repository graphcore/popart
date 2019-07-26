#include <memory>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op.hpp>
#include <popart/op/ipucopy.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>

#include <popart/transforms/interipucopy.hpp>

namespace popart {

class CopiedTensors {

  // record which tensors have been copied to which ipu's
  std::map<TensorId, std::vector<IpuNumber>> tensorMap;

public:
  CopiedTensors() = default;

  // Return true if the tensor has been copied to the ipu
  bool find(TensorId id, IpuNumber ipuNumber) {
    bool found = false;
    auto it    = tensorMap.find(id);
    if (it != tensorMap.end()) {
      auto it2 = std::find(it->second.begin(), it->second.end(), ipuNumber);
      if (it2 != it->second.end()) {
        found = true;
      }
    }

    return found;
  }

  // Record that tensor (id) has been copied to ipuNumber
  void add(TensorId id, IpuNumber ipuNumber) {
    auto it = tensorMap.find(id);
    if (it != tensorMap.end()) {
      it->second.push_back(ipuNumber);
    } else {
      tensorMap.insert(std::make_pair(id, std::vector<IpuNumber>({ipuNumber})));
    }
  }

  void clear() { tensorMap.clear(); }
};

std::size_t InterIpuCopy::id() { return typeid(InterIpuCopy).hash_code(); }

TensorId InterIpuCopy::generateCopiedTensorId(Tensor *tensor,
                                              int64_t toIpu) const {
  // The copiedTensor id needs to be unique as the same tensor may be copied to
  // multiple ipus's
  TensorId copiedTensor = tensor->id + "_c" + std::to_string(toIpu);
  return copiedTensor;
}

void InterIpuCopy::connectIpuCopy(Graph &,
                                  Tensor *tensor,
                                  Op *fromOp,
                                  int64_t fromIpu,
                                  Op *toOp,
                                  int64_t toIpu) const {

  // We have already copied this tensor but we still need to
  // update the 'to' op to use the copied tensor
  logging::transform::debug("Already copied output tensor of {}:{} from "
                            "ipu {} to ipu {}",
                            fromOp->debugName(),
                            tensor->id,
                            fromIpu,
                            toIpu);

  // Copy the list of index's this input tensor is mapped
  auto indices = toOp->input->indices(tensor);

  // Remove this input tensor from the to op for each index
  for (auto i : indices) {
    logging::transform::debug(
        "Disconnecting out {} from {}:{}", tensor->id, toOp->debugName(), i);
    toOp->disconnectInTensor(i, tensor);
  }

  // The copiedTensor id needs to be unique as the same tensor may be copied to
  // multiple ipus's
  TensorId copiedTensor = generateCopiedTensorId(tensor, toIpu);

  // Add the copied input tensor to the to op for each index
  for (auto i : indices) {
    logging::transform::debug(
        "Connecting in {} from {}:{}", copiedTensor, toOp->debugName(), i);
    toOp->connectInTensor(i, copiedTensor);
  }
}

void InterIpuCopy::insertIpuCopy(Graph &graph,
                                 Tensor *tensor,
                                 Op *fromOp,
                                 int64_t fromIpu,
                                 Op *toOp,
                                 int64_t toIpu) const {
  // Need to copy the tensor between ipu's
  logging::transform::debug(
      "Adding copy of output tensor of {}:{} from ipu {} to ipu {}",
      fromOp->debugName(),
      tensor->id,
      fromIpu,
      toIpu);

  Op::Settings settings(graph, "");

  auto ipuCopy_op = std::make_unique<IpuCopyOp>(
      Onnx::CustomOperators::IpuCopy, toIpu, settings);

  auto ipuCopy = ipuCopy_op.get();
  graph.moveIntoGraph(std::move(ipuCopy_op));

  // Copy the list of index's this input tensor is mapped
  auto indices = toOp->input->indices(tensor);

  // Remove this input tensor from the to op for each index
  for (auto i : indices) {
    logging::transform::debug(
        "Disconnecting in {} from {}:{}", tensor->id, toOp->debugName(), i);
    toOp->disconnectInTensor(i, tensor);
  }

  ipuCopy->connectInTensor(0, tensor->id, fromIpu);

  // The copiedTensor id needs to be unique as the same tensor may be copied to
  // multiple ipus's
  TensorId copiedTensor = generateCopiedTensorId(tensor, toIpu);

  ipuCopy->createAndConnectOutTensor(0, copiedTensor);
  ipuCopy->setup();

  // Add the copied input tensor to the to op for each index
  for (auto i : indices) {
    logging::transform::debug(
        "Connecting in {} to {}:{}", copiedTensor, toOp->debugName(), i);
    toOp->connectInTensor(i, copiedTensor);
  }
}

bool InterIpuCopy::apply(Graph &graph) const {
  // If the first op does not have an ipuNumber attribute, assume that no op's
  // have the ipuNumber set and so there is no inter ipu copy required.
  if (graph.getOps().size() > 0 &&
      !(graph.getOps().begin()->second->hasVirtualGraphId())) {
    return false;
  }

  // Keep a record of which tensors have been copied to which ipu's so we don't
  // duplicate a copy of a tensor between ipus
  CopiedTensors copiedTensors;

  // Keep a record of which stream tensors are going to which ops
  std::map<TensorId, std::vector<Op *>> streamsMap;

  // For each op
  for (auto &entry : graph.getOps()) {

    Op *from = entry.second.get();

    if (from->opid != Onnx::CustomOperators::IpuCopy) {

      // Get which ipu the from op is on
      int64_t fromIpu = -1;
      if (from->hasVirtualGraphId()) {
        fromIpu = from->getVirtualGraphId();
      }

      // For each input tensor
      auto &input = from->input;
      for (auto &t : input->tensorMap()) {
        Tensor *tensor = t.second;

        // Record the tensors so we can later work out if any input
        // tensor is going to two ipus
        if (tensor->tensorType() == TensorType::Stream ||
            tensor->tensorType() == TensorType::Const ||
            tensor->tensorType() == TensorType::Variable) {
          auto it = streamsMap.find(tensor->id);
          if (it == streamsMap.end()) {
            std::vector<Op *> streams = {from};
            streamsMap.insert(std::make_pair(tensor->id, streams));
          } else {
            streamsMap[tensor->id].push_back(from);
          }
        }
      }

      // For each output tensor
      auto &output = from->output;
      for (auto &t : output->tensorMap()) {

        Tensor *tensor = t.second;

        // For each consumer op of the tensor
        // but, take a copy of the map as we will be modifying it.
        std::map<Op *, int> map = t.second->consumers.getMap();
        for (auto &c : map) {

          Op *to = c.first;

          if (to->opid != Onnx::CustomOperators::IpuCopy) {

            // Get which ipu the to op is on
            int64_t toIpu = -1;
            if (to->hasVirtualGraphId()) {
              toIpu = to->getVirtualGraphId();
            }

            // If the ops are not on the same ipu
            if (fromIpu != toIpu) {

              bool alreadyCopied = copiedTensors.find(tensor->id, toIpu);

              if (alreadyCopied == true) {
                connectIpuCopy(graph, tensor, from, fromIpu, to, toIpu);
              } else {
                insertIpuCopy(graph, tensor, from, fromIpu, to, toIpu);

                // Record the copy
                copiedTensors.add(tensor->id, toIpu);
              }
            }
          }
        }
      }
    }
  }

  // For any stream tensors that are mapped to multiple ipus we will
  // use the first ipu the list as the input from the host and then
  // add ops in copy that tensor to other ipus
  copiedTensors.clear();
  for (auto &s : streamsMap) {

    if (s.second.size() > 1) {

      auto sourceOp = s.second[0];

      // Get which ipu the to op is on
      int64_t sourceIpu = -1;
      if (sourceOp->hasVirtualGraphId()) {
        sourceIpu = sourceOp->getVirtualGraphId();
      }

      for (auto &op : s.second) {

        int64_t toIpu = -1;
        if (op->hasVirtualGraphId()) {
          toIpu = op->getVirtualGraphId();
        }

        // It the case of the first op the ipu will be the same so nothing to do
        if (sourceIpu != toIpu) {

          logging::transform::debug(
              "Adding op to copy streaming tensor {} from ipu {} to ipu {}",
              s.first,
              sourceIpu,
              toIpu);

          Tensor *tensor = graph.getTensors().get(s.first);

          bool alreadyCopied = copiedTensors.find(tensor->id, toIpu);
          if (alreadyCopied == true) {
            connectIpuCopy(graph, tensor, sourceOp, sourceIpu, op, toIpu);
          } else {
            insertIpuCopy(graph, tensor, sourceOp, sourceIpu, op, toIpu);

            // Record the copy
            copiedTensors.add(tensor->id, toIpu);
          }
        }
      }
    }
  }

  return true;
}

namespace {
bool init = Transform::registerTransform(new InterIpuCopy);
}

} // namespace popart
