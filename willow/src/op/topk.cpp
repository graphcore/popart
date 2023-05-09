// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <popart/graph.hpp>
#include <popart/op/topk.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>

#include "popart/attributes.hpp"
#include "popart/datatype.hpp"
#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/op/basesort.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/tensorinfo.hpp"

namespace popart {

TopKOp::TopKOp(const OperatorIdentifier &opid_,
               int64_t K_,
               int64_t axis_,
               bool largest_,
               bool sorted_,
               bool stable_,
               const Op::Settings &settings_,
               const nonstd::optional<float> &available_memory_proportion_)
    : BaseSortOp(opid_, axis_, settings_), K(K_), largest(largest_),
      sorted(sorted_), stable(stable_),
      available_memory_proportion(available_memory_proportion_) {}

TopKOp::TopKOp(const OperatorIdentifier &opid_,
               int64_t K_,
               int64_t axis_,
               bool largest_,
               bool sorted_,
               const Op::Settings &settings_,
               const nonstd::optional<float> &available_memory_proportion_)
    : TopKOp(opid_,
             K_,
             axis_,
             largest_,
             sorted_,
             false /*stable*/,
             settings_,
             available_memory_proportion_) {}

std::unique_ptr<Op> TopKOp::clone() const {
  return std::make_unique<TopKOp>(*this);
}

void TopKOp::setup() {

  validateAxis();

  auto shape = inShape(getInIndex());
  if (shape.at(getAxis()) < getK()) {
    throw error("Cannot take top-{} on dim of size {}, invalid Op {}",
                getK(),
                getAxis(),
                str());
  }

  shape[getAxis()] = getK();

  outInfo(getIndicesOutIndex()) = TensorInfo(DataType::INT32, shape);
  outInfo(getValuesOutIndex()) =
      TensorInfo(inInfo(getInIndex()).dataType(), shape);
}

void TopKOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  BaseSortOp::appendOutlineAttributes(os);

  if (opid.version == 1) {
    os.appendAttribute("K", K);
  } else {
    // Append the determined K so that this op may be used
    // in outlining
    os.appendAttribute("_K", K);
  }

  os.appendAttribute(sAvailMemAttribute, available_memory_proportion);
}

int64_t TopKOp::getK() const noexcept { return K; }
bool TopKOp::getLargest() const noexcept { return largest; }
bool TopKOp::getSorted() const noexcept { return sorted; }
bool TopKOp::getStable() const noexcept { return stable; }

std::vector<std::unique_ptr<Op>> TopKOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> result;
  result.push_back(std::make_unique<TopKGradOp>(*this));
  return result;
}

TopKGradOp::TopKGradOp(const TopKOp &topk)
    : Op(Onnx::GradOperators::TopKGrad, topk.getSettings()),
      axis(topk.getAxis()), gradOutInfo(topk.inInfo(BaseSortOp::getInIndex())),
      available_memory_proportion(topk.getAvailableMemoryProportion()) {}

std::unique_ptr<Op> TopKGradOp::clone() const {
  return std::make_unique<TopKGradOp>(*this);
}

const std::vector<GradInOutMapper> &TopKGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      // gradient of the TopK values output:
      {gradInIndex(), TopKOp::getValuesOutIndex(), GradOpInType::GradOut},
      // The indices output of the TopK Op:
      {indicesInIndex(), TopKOp::getIndicesOutIndex(), GradOpInType::Out}};
  return inInfo;
}

const std::map<int, int> &TopKGradOp::gradOutToNonGradIn() const {

  static const std::map<int, int> outInfo = {
      {gradOutIndex(), BaseSortOp::getInIndex()}};

  return outInfo;
}

void TopKGradOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("axis", axis);
  os.appendAttribute(sAvailMemAttribute, available_memory_proportion);
}

int64_t TopKGradOp::getAxis() const { return axis; }

const TensorInfo &TopKGradOp::getGradOutInfo() const { return gradOutInfo; }

void TopKGradOp::setup() { outInfo(gradOutIndex()) = gradOutInfo; }

namespace {
Op *topKFactory(const OpCreatorInfo &info, Graph &graph) {
  // largest is optional
  const bool largest = checkedIntToBool(
      info.attributes.getAttribute<Attributes::Int>("largest", 1));
  // sorted is optional
  const bool sorted = checkedIntToBool(
      info.attributes.getAttribute<Attributes::Int>("sorted", 1));

  // axis is optional
  const int64_t axis =
      info.attributes.getAttribute<Attributes::Int>("axis", -1);

  int64_t K = 0;
  if (info.opid.version == 1) {
    // k is required, so has no default value.
    K = info.attributes.getAttribute<Attributes::Int>("k");
  } else if (info.opid.version == 10 || info.opid.version == 11) {
    static constexpr int kInIndex = 1;
    K                             = info.getInputScalarValue<int64_t>(kInIndex);
  } else {
    throw error("Unsupported operator version {} for topK", info.opid.version);
  }

  nonstd::optional<float> available_memory_proportion;

  if (info.attributes.hasAttribute(sAvailMemAttribute)) {
    available_memory_proportion =
        info.attributes.getAttribute<Attributes::Float>(sAvailMemAttribute);
  }

  Op *op = graph.createOp<TopKOp>(info.opid,
                                  K,
                                  axis,
                                  largest,
                                  sorted,
                                  info.settings,
                                  available_memory_proportion);

  // Connect only the first input.
  op->connectInTensor(TopKOp::getInIndex(), info.getInputIds().at(0));
  op->createAndConnectOutTensor(TopKOp::getValuesOutIndex(),
                                info.getOutputIds().at(0));
  op->createAndConnectOutTensor(TopKOp::getIndicesOutIndex(),
                                info.getOutputIds().at(1));

  return op;
}

static OpDefinition::DataTypes T_V1 = {DataType::FLOAT16, DataType::FLOAT};

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
    topKOpV1Def({OpDefinition::Inputs({{"X", T_V1}}),
                 OpDefinition::Outputs({{"Values", T}, {"Indicies", K}}),
                 OpDefinition::Attributes({
                     {"axis", {"*"}},
                     {"k", {"*"}},
                 })});

static OpDefinition
    topKOpDef({OpDefinition::Inputs({
                   {"X", T},
                   {"K", K, true},
               }),
               OpDefinition::Outputs({{"Values", T}, {"Indicies", K}}),
               OpDefinition::Attributes({
                   {"axis", {"*"}},
               })});

static OpCreator<TopKOp>
    TopKOpCreator(OpDefinitions({
                      {Onnx::Operators::TopK_1, topKOpV1Def},
                      {Onnx::Operators::TopK_10, topKOpDef},
                      {Onnx::Operators::TopK_11, topKOpDef},
                  }),
                  topKFactory,
                  true);
} // namespace

} // namespace popart
