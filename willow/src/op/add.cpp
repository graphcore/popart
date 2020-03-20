// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <vector>
// for `find', we need the algorithm header
#include <algorithm>
#include <memory>
#include <popart/op/add.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

template <typename T>
static std::vector<T>
padShape(const std::vector<T> &shape, int padded_size, T pad_value) {
  std::vector<T> result(padded_size - shape.size(), pad_value);
  result.insert(result.end(), shape.begin(), shape.end());
  return result;
}

template <typename T>
static std::vector<T> unpadShape(const std::vector<T> &shape,
                                 int unpadded_size) {
  std::vector<T> result;
  auto offset = shape.size() - unpadded_size;
  result.insert(result.begin(), shape.begin() + offset, shape.end());
  return result;
}

// TODO : T6250 : Add support for V6 axis & broadcast attributes

AddOp::AddOp(const OperatorIdentifier &_opid, const Op::Settings &settings_)
    : ElementWiseBinaryOp(_opid, settings_) {

  // TODO : Use the attributes in Add-6
}

std::unique_ptr<Op> AddOp::clone() const {
  return std::make_unique<AddOp>(*this);
}

std::vector<std::unique_ptr<Op>> AddOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;

  const auto &shape_a0 = inShape(getArg0InIndex());
  const auto &shape_a1 = inShape(getArg1InIndex());
  const auto &shape_o0 = outShape(getOutIndex());

  upops.emplace_back(std::make_unique<AddArg0GradOp>(
      *this, npReductionAxis(shape_a0, shape_o0)));
  upops.emplace_back(std::make_unique<AddArg1GradOp>(
      *this, npReductionAxis(shape_a1, shape_o0)));

  return upops;
}

std::vector<std::tuple<OperatorIdentifier, float>>
AddOp::inplacePriorityDefault() const {
  auto outSize  = outInfo(AddOp::getOutIndex()).nelms();
  auto arg0Size = inInfo(AddOp::getArg0InIndex()).nelms();
  auto arg1Size = inInfo(AddOp::getArg1InIndex()).nelms();

  std::vector<std::tuple<OperatorIdentifier, float>> result;

  if (outSize == arg0Size) {
    auto lhsPriority =
        defaultInplacePriorities.at(Onnx::CustomOperators::AddLhsInplace);
    result.push_back({Onnx::CustomOperators::AddLhsInplace, lhsPriority});
  }
  if (outSize == arg1Size) {
    auto rhsPriority =
        defaultInplacePriorities.at(Onnx::CustomOperators::AddRhsInplace);
    result.push_back({Onnx::CustomOperators::AddRhsInplace, rhsPriority});
  }

  return result;
}

std::unique_ptr<Op>
AddOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::AddLhsInplace) {
    return std::make_unique<AddLhsInplaceOp>(*this);
  } else if (operator_id == Onnx::CustomOperators::AddRhsInplace) {
    return std::make_unique<AddRhsInplaceOp>(*this);
  }

  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

void AddOp::setInplacePriority(const OperatorIdentifier &x, float p) {
  defaultInplacePriorities[x] = p;
}

AddLhsInplaceOp::AddLhsInplaceOp(const AddOp &op)
    : AddOp(Onnx::CustomOperators::AddLhsInplace, op.getSettings()) {}

AddLhsInplaceOp::AddLhsInplaceOp(const Op::Settings &settings_)
    : AddOp(Onnx::CustomOperators::AddLhsInplace, settings_) {}

std::unique_ptr<Op> AddLhsInplaceOp::clone() const {
  return std::make_unique<AddLhsInplaceOp>(*this);
}

view::Regions AddLhsInplaceOp::modifies(InIndex index) const {
  if (index == getArg0InIndex()) {
    return {view::Region::getFull(inShape(index))};
  } else if (index == getArg1InIndex()) {
    return {view::Region::getEmpty(inRank(index))};
  } else {
    throw error("Invalid index passed to AddLhsInplaceOp::modifies");
  }
}

view::Regions AddLhsInplaceOp::aliases(InIndex in, OutIndex) const {
  if (in == getArg0InIndex()) {
    return {view::Region::getFull(inShape(in))};
  } else if (in == getArg1InIndex()) {
    return {view::Region::getEmpty(inRank(in))};
  } else {
    throw error("Invalid index passed to AddLhsInplaceOp::modifies");
  }
}

AddRhsInplaceOp::AddRhsInplaceOp(const AddOp &op)
    : AddOp(Onnx::CustomOperators::AddRhsInplace, op.getSettings()) {}

std::unique_ptr<Op> AddRhsInplaceOp::clone() const {
  return std::make_unique<AddRhsInplaceOp>(*this);
}

view::Regions AddRhsInplaceOp::modifies(InIndex index) const {
  if (index == getArg0InIndex()) {
    return {view::Region::getEmpty(inRank(index))};
  } else if (index == getArg1InIndex()) {
    return {view::Region::getFull(inShape(index))};
  } else {
    throw error("Invalid index passed to AddRhsInplaceOp::modifies");
  }
}

view::Regions AddRhsInplaceOp::aliases(InIndex in, OutIndex) const {
  if (in == getArg0InIndex()) {
    return {view::Region::getEmpty(inRank(in))};
  } else if (in == getArg1InIndex()) {
    return {view::Region::getFull(inShape(in))};
  } else {
    throw error("Invalid index passed to AddRhsInplaceOp::modifies");
  }
}

view::RegMap AddOp::fwdRegMap(InIndex argIndex, OutIndex) const {

  auto out_shape = outShape(AddOp::getOutIndex());
  auto in_shape  = inShape(argIndex);
  return [out_shape, in_shape](const view::Region &r) {
    auto out_size  = static_cast<int>(out_shape.size());
    auto arg_shape = padShape(in_shape, out_size, int64_t{1});
    auto lower     = padShape(r.getLower(), out_size, int64_t{0});
    auto upper     = padShape(r.getUpper(), out_size, int64_t{1});

    // broadcasting
    for (int i = 0; i < out_shape.size(); i++) {
      if (arg_shape[i] == 1 && out_shape[i] > 1) {
        upper[i] = out_shape[i];
      }
    }

    return view::Regions(1, view::Region{lower, upper});
  };
}

view::RegMap AddOp::bwdRegMap(InIndex argIndex, OutIndex) const {

  auto arg_shape = inShape(argIndex);
  auto arg_size  = static_cast<int>(arg_shape.size());
  auto out_shape = unpadShape(outShape(AddOp::getOutIndex()), arg_size);
  return [arg_size, out_shape, arg_shape](const view::Region &r) {
    auto lower = unpadShape(r.getLower(), arg_size);
    auto upper = unpadShape(r.getUpper(), arg_size);

    // unbroadcasting
    for (int i = 0; i < out_shape.size(); i++) {
      if (arg_shape[i] == 1 && out_shape[i] > 1) {
        lower[i] = 0;
        upper[i] = 1;
      }
    }

    return view::Regions(1, view::Region{lower, upper});
  };
}

AddArg0GradOp::AddArg0GradOp(const AddOp &op_,
                             const std::vector<int64_t> &axes_)
    : ReduceSumOp(Onnx::GradOperators::AddArg0Grad,
                  axes_,
                  false,
                  op_.getSettings()),
      forward_op_arg_info(op_.inInfo(AddOp::getArg0InIndex())) {}

const std::map<int, int> &AddArg0GradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), AddOp::getArg0InIndex()}};
  return outInfo;
}

const std::vector<GradInOutMapper> &AddArg0GradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), AddOp::getOutIndex(), GradOpInType::GRADOUT}};
  return inInfo;
}

void AddArg0GradOp::setup() { outInfo(getOutIndex()) = forward_op_arg_info; }

AddArg1GradOp::AddArg1GradOp(const AddOp &op_,
                             const std::vector<int64_t> &axes_)
    : ReduceSumOp(Onnx::GradOperators::AddArg1Grad,
                  axes_,
                  false,
                  op_.getSettings()),
      forward_op_arg_info(op_.inInfo(AddOp::getArg1InIndex())) {}

const std::map<int, int> &AddArg1GradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), AddOp::getArg1InIndex()}};
  return outInfo;
}

const std::vector<GradInOutMapper> &AddArg1GradOp::gradInputInfo() const {
  // input at index 0 : gradient of output of add
  // might need to reduce across certain axes of this
  // if numpy broadcasting happened
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), AddOp::getOutIndex(), GradOpInType::GRADOUT}};
  return inInfo;
}

void AddArg1GradOp::setup() { outInfo(getOutIndex()) = forward_op_arg_info; }

namespace {

static OpDefinition::DataTypes T = {DataType::UINT32,
                                    DataType::UINT64,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT16,
                                    DataType::FLOAT};

static OpDefinition addOpDef({OpDefinition::Inputs({{"A", T}, {"B", T}}),
                              OpDefinition::Outputs({{"C", T}}),
                              OpDefinition::Attributes({})});

static OpCreator<AddOp> addOpCreator(OpDefinitions(
    {{Onnx::Operators::Add_6, addOpDef}, {Onnx::Operators::Add_7, addOpDef}}));

} // namespace

} // namespace popart
