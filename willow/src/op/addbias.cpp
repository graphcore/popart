// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <aliasmodel.hpp>
#include <memory>
#include <numeric>
#include <popart/op/addbias.hpp>
#include <popart/op/conv.hpp>
#include <popart/opmanager.hpp>
#include <popart/tensor.hpp>

namespace popart {

AddBiasOp::AddBiasOp(const OperatorIdentifier &_opid,
                     const Op::Settings &settings_)
    : Op(_opid, settings_) {}

std::unique_ptr<Op> AddBiasOp::clone() const {
  return std::make_unique<AddBiasOp>(*this);
}

std::vector<std::unique_ptr<Op>> AddBiasOp::getGradOps() {
  std::vector<std::unique_ptr<Op>> upops;
  upops.emplace_back(std::make_unique<AddBiasDataGradOp>(*this));

  // "Borrowed" from poplin::convolutionBiasUpdate
  std::vector<int64_t> reduceDims(outRank(getOutIndex()) - 1);
  std::iota(reduceDims.begin() + 1, reduceDims.end(), 2);

  upops.emplace_back(std::make_unique<AddBiasBiasGradOp>(*this, reduceDims));
  return upops;
}

void AddBiasOp::setup() { outInfo(getOutIndex()) = inInfo(getDataInIndex()); }

std::vector<std::tuple<OperatorIdentifier, float>>
AddBiasOp::inplacePriorityDefault() const {
  return {{Onnx::CustomOperators::AddBiasInplace, 10}};
}

std::unique_ptr<Op>
AddBiasOp::getInplaceVariant(const OperatorIdentifier &operator_id) const {
  if (operator_id == Onnx::CustomOperators::AddBiasInplace) {
    return std::make_unique<AddBiasInplaceOp>(*this);
  }
  // catch remaining cases and throw an error
  return Op::getInplaceVariant(operator_id);
}

view::RegMap AddBiasOp::fwdRegMap(InIndex argIndex, OutIndex) const {
  if (argIndex == getDataInIndex()) {
    // data input maps directly to the output
    return [](const view::Region &r) { return view::Regions(1, r); };
  } else if (argIndex == getBiasInIndex()) {
    // e.g. for 2D conv
    // output is shape [_, x, _, _]
    // biasIn is shape [x]
    // the biasIn slice [a:b]
    // maps to the output slice [:, a:b, :, :]
    auto out_shape     = outShape(getOutIndex());
    auto upper_capture = outShape(getOutIndex());
    return [out_shape, upper_capture](const view::Region &r) {
      auto upper = upper_capture;
      // Minimum rank 3: batch_size, output_channels, spatial dim(s)
      if (out_shape.size() < 3) {
        throw error("unexpected output shape for AddBiasInplace");
      }
      if (r.getLower().size() != 1) {
        throw error("unexpected region shape for AddBiasInplace");
      }

      auto lower = std::vector<int64_t>(out_shape.size(), 0);

      lower.at(1) = r.getLower().at(0);
      upper.at(1) = r.getUpper().at(0);

      return view::Regions(1, view::Region{lower, upper});
    };
  } else {
    throw error("Bad index ({}) to AddBiasOp::fwdRegMap", argIndex);
  }
}

view::RegMap AddBiasOp::bwdRegMap(InIndex argIndex, OutIndex) const {
  if (argIndex == getDataInIndex()) {
    return [](const view::Region &r) { return view::Regions(1, r); };
  } else if (argIndex == getBiasInIndex()) {
    // output is shape [_, x, _, _]
    // biasIn is shape [x]
    // the output slice [_, a:b, _, _]
    // maps to the biasIn slice [a:b]
    auto out_shape   = outShape(getOutIndex());
    auto biasInIndex = getBiasInIndex();
    return [biasInIndex, out_shape](const view::Region &r) {
      if (r.getLower().size() != 4) {
        throw error("unexpected region size in AddBiasInplace::bwdRegMap({})",
                    biasInIndex);
      }

      int64_t a = r.getLower().at(1);
      int64_t b = r.getUpper().at(1);

      return view::Regions(1, view::Region{{a}, {b}});
    };
  } else {
    throw error("Bad index ({}) to AddBiasOp::bwdRegMap", argIndex);
  }
}

AddBiasInplaceOp::AddBiasInplaceOp(const AddBiasOp &op)
    : AddBiasOp(Onnx::CustomOperators::AddBiasInplace, op.getSettings()) {}

std::unique_ptr<Op> AddBiasInplaceOp::clone() const {
  return std::make_unique<AddBiasInplaceOp>(*this);
}

std::vector<std::tuple<OperatorIdentifier, float>>
AddBiasInplaceOp::inplacePriorityDefault() const {
  return {};
}

std::unique_ptr<Op>
AddBiasInplaceOp::getInplaceVariant(const OperatorIdentifier &o) const {
  // this throws an error
  return Op::getInplaceVariant(o);
}

view::Regions AddBiasInplaceOp::modifies(InIndex index) const {
  if (index == getDataInIndex()) {
    return {view::Region::getFull(inShape(index))};
  } else if (index == getBiasInIndex()) {
    return {view::Region::getEmpty(inRank(index))};
  } else {
    throw error("Invalid index passed to AddBiasInplaceOp::modifies");
  }
}

view::Regions AddBiasInplaceOp::aliases(InIndex in, OutIndex) const {
  if (in == getDataInIndex()) {
    return {view::Region::getFull(inShape(in))};
  } else if (in == getBiasInIndex()) {
    return {view::Region::getEmpty(inRank(in))};
  } else {
    throw error("Invalid index passed to AddBiasInplaceOp::modifies");
  }
}

void AddBiasOp::growAliasModel(AliasModel &m) const {
  m.insertUnaryModifier0(*this);
}

void AddBiasOp::setProposal(poprithms::memory::inplace::Proposal &proposal,
                            const AliasModel &aliaser,
                            OperatorIdentifier id) const {
  setProposalGate0(proposal, aliaser, id);
}

AddBiasBiasGradOp::AddBiasBiasGradOp(const AddBiasOp &op_,
                                     const std::vector<int64_t> &_axes)
    : ReduceSumOp(Onnx::CustomGradOperators::AddBiasBiasGrad,
                  _axes,
                  0,
                  op_.getSettings()) {}

const std::map<int, int> &AddBiasBiasGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), AddBiasOp::getBiasInIndex()}};

  return outInfo;
}

const std::vector<GradInOutMapper> &AddBiasBiasGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), AddBiasOp::getOutIndex(), GradOpInType::GradOut}};

  return inInfo;
}

AddBiasDataGradOp::AddBiasDataGradOp(const AddBiasOp &op)
    : IdentityOp(Onnx::CustomGradOperators::AddBiasDataGrad, op.getSettings()) {
}

const std::vector<GradInOutMapper> &AddBiasDataGradOp::gradInputInfo() const {
  static const std::vector<GradInOutMapper> inInfo = {
      {getInIndex(), AddBiasOp::getOutIndex(), GradOpInType::GradOut}};

  return inInfo;
}

const std::map<int, int> &AddBiasDataGradOp::gradOutToNonGradIn() const {
  static const std::map<int, int> outInfo = {
      {getOutIndex(), AddBiasOp::getDataInIndex()}};

  return outInfo;
}

namespace {

static OpDefinition::DataTypes T = {DataType::FLOAT16, DataType::FLOAT};

static OpDefinition addBiasOpDef({OpDefinition::Inputs({{"input", T}}),
                                  OpDefinition::Outputs({{"output", T}}),
                                  OpDefinition::Attributes({})});

// The AddBiasOp should not be created from the onnx graph
static OpCreator<AddBiasOp>
    addOpCreator(OpDefinitions({
                     {Onnx::CustomOperators::AddBias, addBiasOpDef},
                 }),
                 false);
} // namespace

} // namespace popart
