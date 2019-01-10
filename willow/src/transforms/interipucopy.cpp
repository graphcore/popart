#include <poponnx/error.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/names.hpp>
#include <poponnx/op.hpp>
#include <poponnx/op/ipucopy.hpp>
#include <poponnx/tensor.hpp>

#include <poponnx/transforms/interipucopy.hpp>

namespace poponnx {

using IpuNumber = int64_t;

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
};

std::size_t InterIpuCopy::id() { return typeid(InterIpuCopy).hash_code(); }

// This function will throw an exception if the attribute is not set
IpuNumber getIpuNumber(const Op *op) {
  IpuNumber num = 0;
  op->nAtts.set(num, sVirtualGraphAttribute);
  return num;
}

bool InterIpuCopy::apply(Ir &ir) const {

  // If the first op does not have an ipuNumber attribute, assume that no op's
  // have the ipuNumber set and so there is no inter ipu copy required.
  if (ir.getOps().size() > 0 && ir.getOps().begin()->second->nAtts.hasAttribute(
                                    sVirtualGraphAttribute) == false) {
    return false;
  }

  // Keep a record of which tensors have been copied to which ipu's so we don't
  // duplicate a copy of a tensor between ipus
  CopiedTensors copiedTensors;

  // For each op
  for (auto &entry : ir.getOps()) {

    Op *from = entry.second.get();

    if (from->opid != Onnx::CustomOperators::IpuCopy) {

      // Get which ipu the from op is on
      int64_t fromIpu = getIpuNumber(from);

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
            int64_t toIpu = getIpuNumber(to);

            // If the ops are not on the same ipu
            if (fromIpu != toIpu) {

              bool alreadyCopied = copiedTensors.find(tensor->id, toIpu);

              if (alreadyCopied == true) {
                // We have already copied this tensor but we still need to
                // update the 'to' op to use the copied tensor
                logging::ir::debug("Already copied output tensor of {}:{} from "
                                   "ipu {} to ipu {}",
                                   from->str(),
                                   tensor->id,
                                   fromIpu,
                                   toIpu);

                // Copy the list of index's this input tensor is mapped
                auto indices = to->input->indices(tensor);

                // Remove this input tensor from the to op for each index
                for (auto i : indices) {
                  logging::ir::debug("Disconnecting out {} from {}:{}",
                                     tensor->id,
                                     to->str(),
                                     i);
                  to->disconnectInTensor(i, tensor);
                }

                TensorId copiedTensor = tensor->id + "_c";

                // Add the copied input tensor to the to op for each index
                for (auto i : indices) {
                  logging::ir::debug("Connecting in {} from {}:{}",
                                     copiedTensor,
                                     to->str(),
                                     i);
                  to->connectInTensor(i, copiedTensor);
                }

              } else {
                // Need to copy the tensor between ipu's
                logging::ir::debug(
                    "Need to copy output tensor of {}:{} from ipu {} to ipu {}",
                    from->str(),
                    tensor->id,
                    fromIpu,
                    toIpu);

                auto ipuCopy_op = make_unique<IpuCopyOp>(
                    Onnx::CustomOperators::IpuCopy, &ir, toIpu);

                auto ipuCopy = ipuCopy_op.get();
                ir.moveIntoIr(std::move(ipuCopy_op));

                // Copy the list of index's this input tensor is mapped
                auto indices = to->input->indices(tensor);

                // Remove this input tensor from the to op for each index
                for (auto i : indices) {
                  logging::ir::debug("Disconnecting out {} from {}:{}",
                                     tensor->id,
                                     to->str(),
                                     i);
                  to->disconnectInTensor(i, tensor);
                }

                ipuCopy->connectInTensor(0, tensor->id);

                TensorId copiedTensor = tensor->id + "_c";

                if (ir.getTensors().contains(copiedTensor) == false) {
                  ipuCopy->createAndConnectOutTensor(0, copiedTensor);
                }

                // Add the copied input tensor to the to op for each index
                for (auto i : indices) {
                  logging::ir::debug("Connecting in {} from {}:{}",
                                     copiedTensor,
                                     to->str(),
                                     i);
                  to->connectInTensor(i, copiedTensor);
                }

                // Record the copy
                copiedTensors.add(tensor->id, toIpu);
              }
            }
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

} // namespace poponnx
