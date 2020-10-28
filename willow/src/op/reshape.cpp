// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
#include <numeric>
#include <onnx/onnx_pb.h>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/reshape.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>

namespace popart {

std::unique_ptr<Op>
ReshapeOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::ReshapeInplace) {
    return std::make_unique<ReshapeInplaceOp>(*this);
  }
  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

view::RegMap ReshapeBaseOp::fwdRegMap(InIndex inIndex,
                                      OutIndex outIndex) const {
  if (inIndex != 0 || outIndex != 0) {
    throw internal_error("[ReshapeBaseOp::fwdRegMap] "
                         "Received input index {} but only 0 allowed, "
                         "This for Op {}, ",
                         inIndex,
                         str());
  }
  auto inRegion    = view::Region::getFull(inInfo(getInIndex()).shape());
  auto outRegion   = view::Region::getFull(outInfo(getOutIndex()).shape());
  auto emptyRegion = view::Region::getEmpty(outRank(getOutIndex()));
  return [emptyRegion, inRegion, outRegion](const view::Region &r) {
    if (r.isEmpty()) {
      return view::Regions(1, emptyRegion);
    }
    return r.reshape(inRegion, outRegion);
  };
}

view::RegMap ReshapeBaseOp::bwdRegMap(InIndex inIndex,
                                      OutIndex outIndex) const {
  if (inIndex != 0 || outIndex != 0) {
    throw internal_error("[ReshapeBaseOp::bwdRegMap] "
                         "Received input index {} but only 0 allowed, "
                         "This for Op {}, ",
                         inIndex,
                         str());
  }
  auto inRegion    = view::Region::getFull(inInfo(getInIndex()).shape());
  auto outRegion   = view::Region::getFull(outInfo(getOutIndex()).shape());
  auto emptyRegion = view::Region::getEmpty(inRank(getInIndex()));
  return [emptyRegion, inRegion, outRegion](const view::Region &r) {
    if (r.isEmpty()) {
      return view::Regions(1, emptyRegion);
    }
    return r.reshape(outRegion, inRegion);
  };
}

ReshapeInplaceOp::ReshapeInplaceOp(const ReshapeOp &op)
    : ReshapeBaseOp(Onnx::CustomOperators::ReshapeInplace,
                    op.getOutShape(),
                    op.settings) {}

ReshapeInplaceOp::ReshapeInplaceOp(const OperatorIdentifier &_opid,
                                   const Shape &shape_,
                                   const Op::Settings &settings_)
    : ReshapeBaseOp(_opid, shape_, settings_) {}

std::unique_ptr<Op> ReshapeInplaceOp::clone() const {
  return std::make_unique<ReshapeInplaceOp>(*this);
}

std::unique_ptr<Op> ReshapeOp::clone() const {
  return std::make_unique<ReshapeOp>(*this);
}

// This will be used by ReshapeGradOp
ReshapeBaseOp::ReshapeBaseOp(const OperatorIdentifier &_opid,
                             const std::vector<int64_t> &ots,
                             const Op::Settings &settings_)
    : Op(_opid, settings_), outShape(ots) {
  finaliseShape();
}

void ReshapeBaseOp::setOutShape(const Shape &value) {
  outShape = value;
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
  upops.emplace_back(std::make_unique<ReshapeGradOp>(*this));
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
    } catch (popart::error &err) {
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
      {getInIndex(), ReshapeOp::getOutIndex(), GradOpInType::GradOut}};
  return inInfo;
}

const std::map<int, int> &ReshapeGradOp::gradOutToNonGradIn() const {
  // the grad-op's output at index 0 corresponds
  // to the non-grad-op's input at index 0
  static const std::map<int, int> outInfo = {
      {getOutIndex(), ReshapeOp::getInIndex()}};
  return outInfo;
}

bool ReshapeBaseOp::canBeReplacedByIdentity() const {
  return inShape(getInIndex()) == outShape;
}

void ReshapeBaseOp::configureShardedOp(Op *const shardOp,
                                       int shardIndex) const {
  Op::configureShardedOp(shardOp, shardIndex);
  if (auto reshape = dynamic_cast<ReshapeBaseOp *>(shardOp)) {
    Shape outShape = reshape->getOutShape();

    auto factor = (inInfo(ReshapeBaseOp::getInIndex()).nelms() /
                   shardOp->inInfo(ReshapeBaseOp::getInIndex()).nelms());

    // TODO T20169: Improve heuristics
    // In general, it is hard to tell where the batch dimension within a reshape
    // is. Here, we assume the batch is always the outer part of the outermost
    // dimension, e.g. a flat tensor of shape [10, 12] at batch size 2
    // is assumed to be [2, 5, 12]
    // This logic will fail if the batch dimension is not included in the
    // outermost dimension larger than the serialization factor.
    for (unsigned i = 0; i < outShape.size(); ++i) {
      if (outShape[i] >= factor) {
        outShape[i] /= factor;
        break;
      }
    }
    reshape->setOutShape(outShape);
  } else {
    throw error(
        "[ReshapeBaseOp] Expected sharded op to be of type ReshapeBaseOp");
  }
}

namespace {

// Can we support more data types?
static OpDefinition::DataTypes T  = {DataType::UINT8,
                                    DataType::UINT16,
                                    DataType::UINT32,
                                    DataType::UINT64,
                                    DataType::INT8,
                                    DataType::INT16,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT16,
                                    DataType::FLOAT,
                                    DataType::BOOL};
static OpDefinition::DataTypes T1 = {DataType::INT64};

static OpDefinition
    reshapeOpDef({OpDefinition::Inputs({{"data", T}, {"shape", T1, true}}),
                  OpDefinition::Outputs({{"output", T}}),
                  OpDefinition::Attributes({})});

static OpCreator<ReshapeOp> reshapeOpCreator(OpDefinitions({
    {Onnx::Operators::Reshape_5, reshapeOpDef},
}));
} // namespace

} // namespace popart
