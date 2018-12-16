#include <onnx/onnx_pb.h>
#include <poponnx/error.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/reshape.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

// This will be used by ReshapeGradOp
ReshapeOp::ReshapeOp(const OpConstructorBundle &bundle,
                     const std::vector<int64_t> &ots)
    : Op(bundle), outShape(ots) {}

ReshapeOp::ReshapeOp(const onnx::NodeProto &node, Ir *_pir) : Op(node, _pir) {
  // we attempt to set outputInfo.

  // The shape tensor should be at index 1
  // check 1 : that there are 2 inputs to this Node
  if (node.input_size() != 2) {
    throw error("Expected input size to ONNX Reshape Operator to be 2");
  }
  TensorId shapeId = node.input(1);

  // check 2 : that there is already a tensor with the shape tensor's name
  if (!_pir->getTensors().contains(shapeId)) {
    throw error("no Tensor named `" + shapeId + "' recorded in Ir. " +
                " This is the second input in the ReshapeOp constructor. ");
  }
  Tensor *shapeTensor = _pir->getTensors().get(shapeId);

  // check 3 : that the tensor has data
  if (!shapeTensor->hasTensorData()) {
    throw error("The shape Tensor `" + shapeId + "' does not have data");
  }
  TensorData *tensorData = shapeTensor->tensorData();

  // check 4 : that the data is int64 (as per the ONNX spec)
  if (shapeTensor->info.dataType() != DataType::INT64) {
    throw error("shape tensor `" + shapeId + "' is not INT64, it is " +
                shapeTensor->info.data_type());
  }

  // check 5 : that is is rank 0 or rank 1
  if (shapeTensor->info.rank() > 1) {
    throw error("new shape tensor should be rank 0/1 in ReshapeOp constructor");
  }

  // Finally, we can set the shape of the output tensor
  outShape      = {};
  int64_t *data = static_cast<int64_t *>(tensorData->data());
  for (int i = 0; i < shapeTensor->info.dim(0); ++i) {
    outShape.push_back(data[i]);
  }
}

const Shape &ReshapeOp::getOutShape() { return outShape; }

std::vector<std::unique_ptr<Op>> ReshapeOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<ReshapeGradOp>(this));
  return upops;
}

std::unique_ptr<Op> ReshapeOp::clone() const {
  return make_unique<ReshapeOp>(*this);
}

void ReshapeOp::setup() {
  // output type  : same as input type;
  // output shape : outShape, determined in the constructor
  outInfo(getOutIndex()) = {inInfo(getInIndex()).dataType(), outShape};

  // sanity check : number of elements unchanged
  auto nOut = outInfo(getOutIndex()).nelms();
  auto nIn  = inInfo(getInIndex()).nelms();
  if (nOut != nIn) {
    std::stringstream ss;
    ss << "Failure in ReshapeOp::setup() for Op " << str() << ". "
       << "The number of elements of the input is " << nIn
       << ", while the number of elements of the output is " << nOut
       << ". The number of elements cannot change for a ReshapeOp";
    throw error(ss.str());
  }
}

void ReshapeOp::connectInTensor(InIndex inIndex, TensorId tenId) {
  // index 0 is the data tensor to be reshaped. We connect
  // the data tensor to this Op as an input, the default connection of
  // an input tensor to its Op
  if (inIndex == 0) {
    defaultConnectInTensor(inIndex, tenId);
  } else if (inIndex == 1) {
    // we let this pass, the reshape Op does not have the second input
    // "shape" tensor which the ONNX Op has, it has already stored
    // the static tensor as a class attribute.
  } else {
    throw error("Unexpected index " + std::to_string(inIndex) +
                " in ReshapeOp::connectInTensor");
  }
}

ReshapeGradOp::ReshapeGradOp(ReshapeOp *op_)
    : ReshapeOp(
          {OpType::RESHAPEGRAD, op_->pir, {}},
          // the output shape of this bwd op is the input shape of the fwd op
          op_->inInfo(ReshapeOp::getInIndex()).shape()) {}

const std::vector<GradInOutMapper> &ReshapeGradOp::gradInputInfo() const {
  // input at index 0 : gradient of output of reshape
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), ReshapeOp::getOutIndex(), GradOpInType::GRADOUT}};
  return inInfo;
}

const std::map<int, int> &ReshapeGradOp::gradOutToNonGradIn() const {
  // the grad-op's output at index 0 corresponds
  // to the non-grad-op's input at index 0
  static const std::map<int, int> outInfo = {
      {getOutIndex(), ReshapeOp::getInIndex()}};
  return outInfo;
}

} // namespace poponnx
