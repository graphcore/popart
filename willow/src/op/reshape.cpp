#include <numeric>
#include <onnx/onnx_pb.h>
#include <poponnx/error.hpp>
#include <poponnx/graph.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op/reshape.hpp>
#include <poponnx/opmanager.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/tensors.hpp>

namespace poponnx {

std::unique_ptr<Op>
ReshapeOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::ReshapeInplace) {
    return make_unique<ReshapeInplaceOp>(*this);
  }
  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

view::RegMap ReshapeBaseOp::fwdRegMap(InIndex inIndex) const {
  if (inIndex != 0) {
    throw error("Internal Logic Error in ReshapeBaseOp::fwdRegMap."
                "Received input index {} but only 0 allowed, "
                "This for Op {}, ",
                inIndex,
                str());
  }
  // being conservative and returning the full region,
  // even for non-full input region :
  auto outRegion   = view::Region::getFull(outInfo(getOutIndex()).shape());
  auto emptyRegion = view::Region::getEmpty(outRank(getOutIndex()));
  return [emptyRegion, outRegion](const view::Region &r) {
    if (r.isEmpty()) {
      return emptyRegion;
    }
    return outRegion;
  };
}

view::RegMap ReshapeBaseOp::bwdRegMap(InIndex inIndex) const {
  if (inIndex != 0) {
    throw error("Internal Logic Error in ReshapeBaseOp::bwdRegMap."
                "Received input index {} but only 0 allowed, "
                "This for Op {}, ",
                inIndex,
                str());
  }
  auto inRegion    = view::Region::getFull(inInfo(getInIndex()).shape());
  auto emptyRegion = view::Region::getEmpty(inRank(getInIndex()));
  return [emptyRegion, inRegion](const view::Region &r) {
    if (r.isEmpty()) {
      return emptyRegion;
    }
    return inRegion;
  };
}

ReshapeInplaceOp::ReshapeInplaceOp(const ReshapeOp &op)
    : ReshapeBaseOp(Onnx::CustomOperators::ReshapeInplace,
                    op.getOutShape(),
                    op.settings) {}

std::unique_ptr<Op> ReshapeInplaceOp::clone() const {
  return make_unique<ReshapeInplaceOp>(*this);
}

std::unique_ptr<Op> ReshapeOp::clone() const {
  return make_unique<ReshapeOp>(*this);
}

// This will be used by ReshapeGradOp
ReshapeBaseOp::ReshapeBaseOp(const OperatorIdentifier &_opid,
                             const std::vector<int64_t> &ots,
                             const Op::Settings &settings_)
    : Op(_opid, settings_), outShape(ots) {
  finaliseShape();
}

const Shape &ReshapeBaseOp::getOutShape() const { return outShape; }

void ReshapeBaseOp::finaliseShape() {
  // replace zeros with size of input dimension
  for (int i = 0; i < outShape.size(); i++) {
    if (outShape[i] == 0) {
      outShape[i] = inShape(getInIndex())[i];
    }
  }

  // a single dimension set to -1 may be inferred
  auto infer_dim = std::find(outShape.begin(), outShape.end(), -1);
  if (infer_dim != outShape.end()) {
    auto in_size  = inInfo(getInIndex()).nelms();
    auto out_size = -std::accumulate(
        outShape.begin(), outShape.end(), 1, std::multiplies<int64_t>());
    *infer_dim = in_size / out_size;

    // search the remaining elements of outShape for another -1
    if (std::find(++infer_dim, outShape.end(), -1) != outShape.end()) {
      throw error("shape input to ReshapeOp can only use -1 to specify one "
                  "unknown dimension");
    }
  }
}

std::vector<std::unique_ptr<Op>> ReshapeOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(make_unique<ReshapeGradOp>(*this));
  return upops;
}

void ReshapeBaseOp::setup() {
  // output type  : same as input type;
  // output shape : outShape, determined in the constructor
  outInfo(getOutIndex()) = {inInfo(getInIndex()).dataType(), outShape};

  // sanity check : number of elements unchanged
  auto nOut = outInfo(getOutIndex()).nelms();
  auto nIn  = inInfo(getInIndex()).nelms();
  if (nOut != nIn) {
    std::stringstream ss;
    ss << "Failure in ReshapeOp::setup() for " << debugName() << ". "
       << "The number of elements of the input is " << nIn
       << ", while the number of elements of the output is " << nOut
       << ". The number of elements cannot change for a ReshapeOp";
    throw error(ss.str());
  }
}

void ReshapeBaseOp::connectInTensor(InIndex inIndex, TensorId tenId) {

  // index 0 is the data tensor to be reshaped. We connect
  // the data tensor to this Op as an input, the default connection of
  // an input tensor to its Op
  if (inIndex == 0) {
    Op::connectInTensor(inIndex, tenId);
  } else if (inIndex == 1) {
    // we attempt to set outputInfo
    try {
      getInTensorData(tenId, outShape);
      finaliseShape();
    } catch (poponnx::error &err) {
      throw error("Need the value of the {} input 'shape' to detemine the "
                  "output shape, but was unable because {}",
                  opid,
                  err.what());
    }

  } else {
    throw error("Unexpected index " + std::to_string(inIndex) +
                " in ReshapeOp::connectInTensor");
  }
}

ReshapeGradOp::ReshapeGradOp(const ReshapeOp &op_)
    : ReshapeOp(
          Onnx::GradOperators::ReshapeGrad,
          // the output shape of this bwd op is the input shape of the fwd op
          op_.inInfo(ReshapeOp::getInIndex()).shape(),
          op_.getSettings()) {}

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

bool ReshapeBaseOp::canBeReplacedByIdentity() {
  return inShape(getInIndex()) == outShape;
}

namespace {
static OpCreator<ReshapeOp> reshapeOpCreator(Onnx::Operators::Reshape_5);
} // namespace

} // namespace poponnx
