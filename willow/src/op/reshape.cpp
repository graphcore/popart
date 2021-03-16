// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <memory>
#include <numeric>
#include <onnx/onnx_pb.h>
#include <poprithms/ndarray/shape.hpp>
#include <poprithms/util/printiter.hpp>
#include <popart/error.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/reshape.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>
#include <popart/tensors.hpp>
#include <popart/topocons.hpp>
#include <popart/util.hpp>

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
                             const Op::Settings &settings_,
                             bool handleZero_)
    : Op(_opid, settings_), outShape(ots), handleZero(handleZero_) {
  finaliseShape();
}

void ReshapeBaseOp::setOutShape(const Shape &value) {
  outShape = value;
  finaliseShape();
}

const Shape &ReshapeBaseOp::getOutShape() const { return outShape; }

void ReshapeBaseOp::finaliseShape() {
  auto bad_dim = std::find_if(
      outShape.begin(), outShape.end(), [&](auto i) { return i < -1; });
  if (bad_dim != outShape.end()) {
    throw error(
        "Dimension, {}, of new shape for Reshape cannot be smaller than -1.",
        *bad_dim);
  }

  // Do not inferr, do not replace zeros.
  if (!handleZero) {
    return;
  }

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
    if (in_size % out_size != 0) {
      throw error("Incompatible inferred dimension, not whole number.");
    }
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
                                       const Settings *const settings_) const {
  Op::configureShardedOp(shardOp, settings_);
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

poprithms::ndarray::Shape getShapeToReshape(const OpCreatorInfo &info) {
  return poprithms::ndarray::Shape(info.settings.graph.get()
                                       .getTensors()
                                       .get(info.getInputIds()[0])
                                       ->info.shape());
}

std::vector<int64_t> getAxesAttrib(const OpCreatorInfo &info) {
  return info.attributes.getAttribute<Attributes::Ints>("axes", {});
}

std::vector<int64_t> getShapeAttrib(const OpCreatorInfo &info) {
  return info.attributes.getAttribute<Attributes::Ints>("shape", {});
}

void checkNegativeShape(const std::vector<int64_t> &newShape) {
  auto bad_dim = std::find_if(
      newShape.begin(), newShape.end(), [&](auto i) { return i < 0; });
  if (bad_dim != newShape.end()) {
    throw error("Attribute shape has negative dimension. "
                "Not supported.");
  }
}

// UnSqueeze //
static OpDefinition
    unsqueezeOpDef({OpDefinition::Inputs({{"data", T}}),
                    OpDefinition::Outputs({{"expanded", T}}),
                    OpDefinition::Attributes({{"axes", {"*"}}})});

static OpCreator<ReshapeOp> unsqueezeOpCreator(
    OpDefinitions({
        {Onnx::Operators::Unsqueeze_1, unsqueezeOpDef},
        {Onnx::Operators::Unsqueeze_11, unsqueezeOpDef},
    }),
    [](const OpCreatorInfo &info) {
      const auto inShape  = getShapeToReshape(info);
      const auto axes     = getAxesAttrib(info);
      const auto outShape = inShape.unsqueeze(
          getAxes_u64(axes, axes.size() + inShape.rank_u64()));
      return std::make_unique<ReshapeOp>(
          Onnx::Operators::Reshape_5, outShape.get(), info.settings);
    },
    true);

// Squeeze //
static OpDefinition squeezeOpDef({OpDefinition::Inputs({{"data", T}}),
                                  OpDefinition::Outputs({{"squeezed", T}}),
                                  OpDefinition::Attributes({{"axes", {"*"}}})});

static OpCreator<ReshapeOp> squeezeOpCreator(
    OpDefinitions({
        {Onnx::Operators::Squeeze_1, squeezeOpDef},
        {Onnx::Operators::Squeeze_11, squeezeOpDef},
    }),
    [](const OpCreatorInfo &info) {
      const auto inShape = getShapeToReshape(info);
      const auto axes    = getAxesAttrib(info);
      const auto outShape =
          axes.empty() ? inShape.squeeze()
                       : inShape.squeeze(getAxes_u64(axes, inShape.rank_u64()));
      return std::make_unique<ReshapeOp>(
          Onnx::Operators::Reshape_5, outShape.get(), info.settings);
    },
    true);

// Flatten //
// ------- //
static OpDefinition flattenOpDef({OpDefinition::Inputs({{"input", T}}),
                                  OpDefinition::Outputs({{"output", T}}),
                                  OpDefinition::Attributes({{"axis", {"*"}}})});

static std::unique_ptr<Op> flattenOpFactory(const OpCreatorInfo &info) {
  const auto inShape = getShapeToReshape(info);
  int64_t axis       = info.attributes.getAttribute<Attributes::Int>("axis", 1);
  axis += (axis < 0 ? inShape.rank_i64() : 0);
  if (axis < 0 || axis > inShape.rank_u64()) {
    throw error("invalid axis {} in flattenOpFactory", axis);
  }
  return std::make_unique<ReshapeOp>(
      Onnx::Operators::Reshape_5,
      inShape.flattenTo2d(static_cast<uint64_t>(axis)).get(),
      info.settings);
}

static OpCreator<ReshapeOp>
    flattenOpCreator({{Onnx::Operators::Flatten_1, flattenOpDef},
                      {Onnx::Operators::Flatten_9, flattenOpDef},
                      {Onnx::Operators::Flatten_11, flattenOpDef}},
                     flattenOpFactory,
                     true);

// Custom reshape with 1 input and shape attrib.
static OpDefinition
    reshape1OpDef({OpDefinition::Inputs({{"data", T}}),
                   OpDefinition::Outputs({{"reshaped", T}}),
                   OpDefinition::Attributes({{"shape", {"*"}}})});

static OpCreator<ReshapeOp> reshape1OpCreator(
    OpDefinitions({{Onnx::CustomOperators::Reshape_1, reshape1OpDef}}),
    [](const OpCreatorInfo &info) {
      const auto outShape = getShapeAttrib(info);
      checkNegativeShape(outShape);
      return std::make_unique<ReshapeOp>(
          Onnx::Operators::Reshape_5, outShape, info.settings, false);
    },
    true);

} // namespace

} // namespace popart
