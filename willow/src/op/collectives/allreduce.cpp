// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <popart/op.hpp>
#include <popart/op/collectives/allreduce.hpp>
#include <popart/op/collectives/collectives.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensorindex.hpp>

#include "popart/attributes.hpp"
#include "popart/datatype.hpp"
#include "popart/error.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/logging.hpp"
#include "popart/names.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/tensorlocation.hpp"

namespace popart {
struct OperatorIdentifier;

AllReduceOp::AllReduceOp(const OperatorIdentifier &_opid,
                         const CollectiveOperator op_,
                         const std::vector<int64_t> ipus_,
                         const Op::Settings &settings_)
    : Op(_opid, settings_), reduceOp(op_), ipus(ipus_) {}

AllReduceOp::AllReduceOp(const OperatorIdentifier &_opid,
                         const CollectiveOperator op_,
                         const std::vector<int64_t> ipus_,
                         const bool identicalInputs_,
                         const bool identicalGradInputs_,
                         const Op::Settings &settings_)
    : Op(_opid, settings_), reduceOp(op_), ipus(ipus_),
      identicalInputs(identicalInputs_),
      identicalGradInputs(identicalGradInputs_) {}

std::unique_ptr<Op> AllReduceOp::clone() const {
  return std::make_unique<AllReduceOp>(*this);
}

std::vector<std::unique_ptr<Op>> AllReduceOp::getGradOps() {
  if (reduceOp != CollectiveOperator::Add) {
    throw error("Cannot create grad op for AllReduceOp. "
                "CollectiveOperator::Add is the only collective opperator "
                "that is currently implemented.");
  }
  std::vector<std::unique_ptr<Op>> result;
  // Reverse identicalInputs <-> identicalGradInputs
  result.push_back(std::make_unique<AllReduceGradOp>(
      reduceOp, ipus, identicalGradInputs, identicalInputs, settings));
  return result;
}

void AllReduceOp::setup() {
  auto numInputs  = input->n();
  auto numOutputs = output->n();

  if (numInputs == 0) {
    throw error("AllReduceOp::setup there must be at least one input.");
  }

  if (numInputs != numOutputs) {
    throw error("AllReduceOp::setup number of inputs does not equal number of "
                "outputs.");
  }

  if (numInputs != ipus.size()) {
    throw error("AllReduceOp::setup number of ipus does not equal number of "
                "inputs and outputs.");
  }

  auto tensorInfo0 = inInfo(0);
  for (int i = 1; i < numInputs; i++) {
    if (tensorInfo0 != inInfo(i)) {
      throw error("AllReduceOp::setup not all inputs have the same shape and "
                  "datatype (tensorinfo).");
    }
  }

  for (int i = 0; i < numOutputs; i++) {
    outInfo(i) = tensorInfo0;
  }
}

void AllReduceOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("reduceOp", reduceOp);
  os.appendAttribute("ipus", ipus);
  os.appendAttribute("identicalInputs", identicalInputs);
  os.appendAttribute("identicalGradInputs", identicalGradInputs);
}

bool AllReduceOp::canBeReplacedByIdentity() const {
  return input->n() == 1 && output->n() == 1;
}

VGraphIdAndTileSet
AllReduceOp::getIntrospectionInVirtualGraphId(InIndex index,
                                              std::set<OpId> &visited) const {
  return {ipus.at(index), settings.tileSet};
}

VGraphIdAndTileSet
AllReduceOp::getIntrospectionOutVirtualGraphId(OutIndex index,
                                               std::set<OpId> &visited) const {
  return {ipus.at(index), settings.tileSet};
}

AllReduceGradOp::AllReduceGradOp(CollectiveOperator op_,
                                 std::vector<int64_t> ipus_,
                                 const bool identicalInputs_,
                                 const bool identicalGradInputs_,
                                 const Op::Settings &settings_)
    : AllReduceOp(Onnx::CustomGradOperators::AllReduceGrad,
                  op_,
                  ipus_,
                  identicalInputs_,
                  identicalGradInputs_,
                  settings_) {

  auto numInputs = ipus_.size();

  for (int i = 0; i < numInputs; i++) {
    inGradMap.push_back({i, i, GradOpInType::GradOut});
  }

  for (int i = 0; i < numInputs; i++) {
    outGradMap[i] = i;
  }
}

std::unique_ptr<Op> AllReduceGradOp::clone() const {
  return std::make_unique<AllReduceGradOp>(*this);
}

const std::vector<GradInOutMapper> &AllReduceGradOp::gradInputInfo() const {
  return inGradMap;
}

const std::map<int, int> &AllReduceGradOp::gradOutToNonGradIn() const {
  return outGradMap;
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

static OpDefinition allReduceOpDef({OpDefinition::Inputs({{"inputs", T}}),
                                    OpDefinition::Outputs({{"outputs", T}}),
                                    OpDefinition::Attributes({
                                        {sCollectiveOperator, {"*"}},
                                        {"ipus", {"*"}},
                                        {"identicalInputs", {"*"}},
                                        {"identicalGradInputs", {"*"}},
                                    })});

static OpCreator<AllReduceOp> allReduceOpCreator(
    OpDefinitions({{Onnx::CustomOperators::AllReduce, allReduceOpDef}}),
    [](const OpCreatorInfo &info) {
      CollectiveOperator op = static_cast<CollectiveOperator>(
          info.attributes.getAttribute<Attributes::Int>(
              sCollectiveOperator, static_cast<int>(CollectiveOperator::Add)));

      std::vector<int64_t> ipus = static_cast<std::vector<int64_t>>(
          info.attributes.getAttribute<Attributes::Ints>("ipus"));

      return std::unique_ptr<Op>(
          new AllReduceOp(info.opid, op, ipus, info.settings));
    },
    true);
} // namespace

} // namespace popart
