#include <memory>
#include <popart/graph.hpp>
#include <popart/op/onehot.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>
#include <popart/tensors.hpp>
#include <popart/typefunctor.hpp>

namespace popart {

OnehotOp::OnehotOp(const OperatorIdentifier &_opid,
                   int64_t axis_,
                   const Op::Settings &settings_)
    : Op(_opid, settings_), axis(axis_) {}

std::unique_ptr<Op> OnehotOp::clone() const {
  return std::make_unique<OnehotOp>(*this);
}

void OnehotOp::setup() {

  // The output type is the same as the input values
  auto outputType = inInfo(getValuesInIndex()).dataType();

  // The outshape is the same as the indices plus an additional
  // dimension at the axis for the one-hot representation
  auto outputShape = inInfo(getIndicesInIndex()).shape();

  // Add the additional axis
  if (axis != -1) {
    outputShape.insert(outputShape.begin() + axis, onehotAxisDim);
  } else {
    outputShape.push_back(onehotAxisDim);
  }

  outInfo(getOutIndex()) = TensorInfo(outputType, outputShape);
}

void OnehotOp::connectInTensor(InIndex inIndex, TensorId tenId) {

  if (inIndex == 1) {
    // Determine the dimension of the onehot axis

    TensorId depthId = tenId;

    // check 2 : that there is already a tensor with the shape tensor's name
    if (!getGraph().getTensors().contains(depthId)) {
      throw error("no Tensor named `" + depthId + "' recorded in Ir. " +
                  " This is the second input in the OneHot constructor. ");
    }

    Tensor *depthTensor = getGraph().getTensors().get(depthId);

    // check 3 : that the tensor has data
    if (!depthTensor->hasTensorData()) {
      throw error("The depth Tensor `" + depthId + "' does not have data");
    }

    TensorData *depthTensorData = depthTensor->tensorData();

    // check 5 : that is is rank 0 i.e. a scalar
    if (depthTensor->info.rank() != 0) {
      throw error("The depth tensor should be rank 0 in OneHot");
    }

    onehotAxisDim = typefunctor::get<typefunctor::Int64FromVoid, int64_t>(
        depthTensor->info.dataType(), depthTensorData->data());
  } else {
    defaultConnectInTensor(inIndex, tenId);
  }
}

std::vector<std::unique_ptr<Op>> OnehotOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<OnehotGradOp>(*this));
  return upops;
}

void OnehotOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("axis", axis);
  os.appendAttribute("depth", onehotAxisDim);
}

OnehotGradOp::OnehotGradOp(const OnehotOp &fwdOp_)
    : Op(Onnx::GradOperators::OneHotGrad, fwdOp_.getSettings()),
      axis(fwdOp_.getAxis()),
      outputShape(fwdOp_.inInfo(OnehotOp::getValuesInIndex()).shape()) {}

std::unique_ptr<Op> OnehotGradOp::clone() const {
  return std::make_unique<OnehotGradOp>(*this);
}

void OnehotGradOp::setup() {
  outInfo(getOutIndex()) =
      TensorInfo(inInfo(getGradInIndex()).dataType(), outputShape);
}

const std::vector<GradInOutMapper> &OnehotGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getGradInIndex(), OnehotOp::getOutIndex(), GradOpInType::GRADOUT},
      {getIndicesInIndex(), OnehotOp::getIndicesInIndex(), GradOpInType::IN}};

  return inInfo;
}

const std::map<int, int> &OnehotGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), OnehotOp::getValuesInIndex()}};

  return outInfo;
}

void OnehotGradOp::appendAttributes(OpSerialiserBase &os) const {
  Op::appendAttributes(os);
  os.appendAttribute("axis", axis);
}

namespace {
static OpCreator<OnehotOp> onehotOpCreator(
    {Onnx::Operators::OneHot_9},
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr) -> std::unique_ptr<Op> {
      int64_t axis = attr.getAttribute<Attributes::Int>("axis", -1);

      return std::unique_ptr<Op>(new OnehotOp(_opid, axis, settings));
    },
    true);

} // namespace
} // namespace popart
