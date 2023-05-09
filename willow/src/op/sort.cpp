// Copyright (c) 2023 Graphcore Ltd. All rights reserved.

#include <popart/graph.hpp>
#include <popart/graphcoreoperators.hpp>
#include <popart/op/sort.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensorinfo.hpp>

namespace popart {

SortOp::SortOp(const OperatorIdentifier &opid_,
               int64_t axis_,
               int64_t axis_size_,
               bool descending_,
               bool stable_,
               const Op::Settings &settings_,
               const nonstd::optional<float> &available_memory_proportion_)
    : TopKOp(opid_,
             axis_size_,
             axis_,
             descending_,
             true /*sorted*/,
             stable_,
             settings_,
             available_memory_proportion_) {}

std::unique_ptr<Op> SortOp::clone() const {
  return std::make_unique<SortOp>(*this);
}

namespace {
Op *sortFactory(const OpCreatorInfo &info, Graph &graph) {
  // descending is optional
  const bool descending = checkedIntToBool(
      info.attributes.getAttribute<Attributes::Int>("descending", 0));
  // stable is optional
  const bool stable = checkedIntToBool(
      info.attributes.getAttribute<Attributes::Int>("stable", 0));
  // axis is optional
  const int64_t axis =
      info.attributes.getAttribute<Attributes::Int>("axis", -1);

  const auto &inputTensorInfo = info.getInputTensorInfo(SortOp::getInIndex());

  const auto rank     = inputTensorInfo.rank();
  const auto axisSize = inputTensorInfo.dim(axis >= 0 ? axis : (rank + axis));

  nonstd::optional<float> available_memory_proportion;

  if (info.attributes.hasAttribute(sAvailMemAttribute)) {
    available_memory_proportion =
        info.attributes.getAttribute<Attributes::Float>(sAvailMemAttribute);
  }

  Op *op = graph.createOp<SortOp>(info.opid,
                                  axis,
                                  axisSize,
                                  descending,
                                  stable,
                                  info.settings,
                                  available_memory_proportion);

  // Connect only the first input.
  op->connectInTensor(SortOp::getInIndex(), info.getInputIds().at(0));
  op->createAndConnectOutTensor(SortOp::getValuesOutIndex(),
                                info.getOutputIds().at(0));
  op->createAndConnectOutTensor(SortOp::getIndicesOutIndex(),
                                info.getOutputIds().at(1));

  return op;
}

static OpDefinition::DataTypes T = {DataType::UINT8,
                                    DataType::UINT16,
                                    DataType::UINT32,
                                    DataType::UINT64,
                                    DataType::INT8,
                                    DataType::INT16,
                                    DataType::INT32,
                                    DataType::INT64,
                                    DataType::FLOAT16,
                                    DataType::FLOAT};
static OpDefinition::DataTypes K = {DataType::INT64};

static OpDefinition
    sortOpDef({OpDefinition::Inputs({
                   {"X", T},
               }),
               OpDefinition::Outputs({{"Values", T}, {"Indicies", K}}),
               OpDefinition::Attributes({{"axis", {"*"}},
                                         {"descending", {"*"}},
                                         {"stable", {"*"}}})});

static constexpr bool isPublic = true;

static OpCreator<SortOp>
    SortOpCreator(OpDefinitions({
                      {Onnx::CustomOperators::Sort, sortOpDef},
                  }),
                  sortFactory,
                  isPublic);
} // namespace

} // namespace popart
