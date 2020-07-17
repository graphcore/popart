// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/init.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

InitOp::InitOp(const OperatorIdentifier &_opid,
               const TensorInfo &tensor_info_,
               const TensorType &tensor_type_,
               const InitType &init_type_,
               const Op::Settings &settings_)
    : Op(_opid, settings_), tensor_info(tensor_info_),
      tensor_type(tensor_type_), init_type(init_type_) {}

std::unique_ptr<Op> InitOp::clone() const {
  return std::make_unique<InitOp>(*this);
}

void InitOp::setup() {
  outInfo(getOutIndex()) = tensor_info;
  output->tensor(getOutIndex())->setTensorType(tensor_type);
}

void InitOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("inittype", static_cast<int>(getInitType()));
}

static OpDefinition::DataTypes T = {DataType::FLOAT,
                                    DataType::FLOAT16,
                                    DataType::INT32,
                                    DataType::UINT32};

static OpDefinition initOpDef({OpDefinition::Inputs({}),
                               OpDefinition::Outputs({{"I", T}}),
                               OpDefinition::Attributes({
                                   {"shape", {"*"}},
                                   {"data_type", {"*"}},
                                   {"tensor_type", {"*"}},
                                   {"init_type", {"*"}},
                               })});

static OpCreator<InitOp> initOpCreator(
    OpDefinitions({{Onnx::CustomOperators::Init_1, initOpDef}}),
    [](const OpCreatorInfo &info) {
      std::vector<int64_t> shape =
          info.attributes.getAttribute<Attributes::Ints>("shape");
      DataType data_type = static_cast<DataType>(
          info.attributes.getAttribute<Attributes::Int>("data_type"));
      TensorType tensor_type = static_cast<TensorType>(
          info.attributes.getAttribute<Attributes::Int>("tensor_type"));
      InitType init_type = static_cast<InitType>(
          info.attributes.getAttribute<Attributes::Int>("init_type"));
      TensorInfo tInfo(data_type, shape);
      return std::unique_ptr<InitOp>(
          new InitOp(info.opid, tInfo, tensor_type, init_type, info.settings));
    },
    true);

} // namespace popart
