// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <boost/range/numeric.hpp>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <popart/op/split.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>
#include <popart/tensorindex.hpp>

#include "popart/attributes.hpp"
#include "popart/datatype.hpp"
#include "popart/error.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/op.hpp"
#include "popart/operators.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/tensorlocation.hpp"

namespace popart {

struct OperatorIdentifier;
} // namespace popart

using boost::accumulate;

namespace popart {

SplitOp::SplitOp(const OperatorIdentifier &opid_,
                 int64_t axis_,
                 const std::vector<int64_t> split_,
                 const Op::Settings &settings_)
    : Op(opid_, settings_), axis(axis_), split(split_) {}

void SplitOp::setup() {
  auto numOutputs = output->n();
  auto splitSizes = getSplitSizes();
  if (splitSizes.size() != numOutputs) {
    throw error("Number of outputs does not match number of requested splits");
  }

  auto type  = inInfo(getInIndex()).dataType();
  auto shape = inInfo(getInIndex()).shape();

  if (axis < 0 || axis >= shape.size()) {
    throw error(
        "Axis {} is out of range for tensor with {} dims", axis, shape.size());
  }

  // sum of splitSizes should be equal to axis being split across
  if (accumulate(splitSizes, 0) != shape.at(axis)) {
    throw error("splits {} invalid for dimension of size {}",
                accumulate(splitSizes, 0),
                shape.at(axis));
  }

  for (int i = 0; i < numOutputs; i++) {
    shape[axis] = splitSizes.at(i);
    outInfo(i)  = {type, shape};
  }
}

std::unique_ptr<Op> SplitOp::clone() const {
  return std::make_unique<SplitOp>(*this);
}

std::vector<std::unique_ptr<Op>> SplitOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<SplitGradOp>(*this, getSettings()));
  return upops;
}

std::vector<int64_t> SplitOp::getSplitSizes() const {
  if (split.size() == 0) {
    auto numOutputs    = output->n();
    auto inShape       = inInfo(getInIndex()).shape();
    auto splitAxisSize = inShape.at(axis);

    if ((splitAxisSize / numOutputs) * numOutputs != splitAxisSize) {
      throw error("Tensor {} ({}) may not be split into equally sized parts "
                  "along axis {}",
                  inTensor(getInIndex())->id,
                  inShape,
                  axis);
    }

    return std::vector<int64_t>(numOutputs, splitAxisSize / numOutputs);
  } else {
    return split;
  }
}

SplitGradOp::SplitGradOp(const SplitOp &fwdOp, const Op::Settings &settings_)
    : Op(Onnx::GradOperators::SplitGrad, settings_),
      fwdOpInInfo(fwdOp.inInfo(SplitOp::getInIndex())), axis(fwdOp.getAxis()) {
  for (int i = 0; i < fwdOp.output->n(); i++) {
    gradInInfo.push_back({i, i, GradOpInType::GradOut});
  }

  outInfoMap.insert({getOutIndex(), SplitOp::getInIndex()});
}

void SplitGradOp::setup() { outInfo(getOutIndex()) = fwdOpInInfo; }

std::unique_ptr<Op> SplitGradOp::clone() const {
  return std::make_unique<SplitGradOp>(*this);
}

const std::vector<GradInOutMapper> &SplitGradOp::gradInputInfo() const {
  return gradInInfo;
}

const std::map<int, int> &SplitGradOp::gradOutToNonGradIn() const {
  return outInfoMap;
}

namespace {

static OpDefinition::DataTypes T = {DataType::UINT8,
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

static OpDefinition
    splitOpDef({OpDefinition::Inputs({{"input", T}}),
                OpDefinition::Outputs({{"outputs", T}}),
                OpDefinition::Attributes({{"axis", {"*"}}, {"split", {"*"}}})});

static OpCreator<SplitOp> splitOpCreator(
    OpDefinitions({{Onnx::Operators::Split_2, splitOpDef},
                   {Onnx::Operators::Split_11, splitOpDef}}),
    [](const OpCreatorInfo &info) {
      auto axis  = info.attributes.getAttribute<Attributes::Int>("axis", 0);
      auto split = info.attributes.getAttribute<Attributes::Ints>("split", {});

      if (axis < 0) {
        auto input = info.getInputTensor(0);
        auto rank  = input->info.rank();
        if (abs(axis) > rank) {
          throw error(
              "Split op input tensor {}, rank {}, axis {} is out of range",
              input->str(),
              rank,
              axis);
        }

        logging::trace("Split op input tensor {}, axis {}, rank {}",
                       input->str(),
                       axis,
                       rank);
        axis = axis + rank;
        logging::trace(
            "Split op input tensor {}, converted axis {}", input->str(), axis);
      }
      return std::make_unique<SplitOp>(info.opid, axis, split, info.settings);
    },
    true);
} // namespace

} // namespace popart
