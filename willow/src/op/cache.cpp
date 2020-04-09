// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/cache.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

CacheStoreOp::CacheStoreOp(const OperatorIdentifier &_opid,
                           const Op::Settings &settings_,
                           RemoteBufferId id)
    : Op(_opid, settings_), remotebuffer_id(id) {}

std::unique_ptr<Op> CacheStoreOp::clone() const {
  return std::make_unique<CacheStoreOp>(*this);
}

void CacheStoreOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("bufferid", remotebuffer_id);
}

CacheLoadOp::CacheLoadOp(const OperatorIdentifier &_opid,
                         const Op::Settings &settings_,
                         RemoteBufferId id)
    : Op(_opid, settings_), remotebuffer_id(id) {}

std::unique_ptr<Op> CacheLoadOp::clone() const {
  return std::make_unique<CacheLoadOp>(*this);
}

void CacheLoadOp::appendOutlineAttributes(OpSerialiserBase &os) const {
  Op::appendOutlineAttributes(os);
  os.appendAttribute("bufferid", remotebuffer_id);
}

void CacheLoadOp::setup() {
  outInfo(getCachedTensorOutIndex()) = inInfo(getCachedTensorInIndex());
}

view::Regions CacheLoadOp::modifies(InIndex index) const {
  if (index == getCachedTensorInIndex()) {
    return {view::Region::getFull(inShape(index))};
  } else if (index == getRemoteBufferOffsetInIndex()) {
    return {view::Region::getEmpty(inRank(index))};
  } else {
    throw error("Invalid index passed to CacheLoadOp::modifies");
  }
}

view::Regions CacheLoadOp::aliases(InIndex in, OutIndex) const {
  if (in == getCachedTensorInIndex()) {
    return {view::Region::getFull(inShape(in))};
  } else if (in == getRemoteBufferOffsetInIndex()) {
    return {view::Region::getEmpty(inRank(in))};
  } else {
    throw error("Invalid index passed to CacheLoadOp::aliases");
  }
}

view::RegMap CacheLoadOp::fwdRegMap(InIndex inIndex, OutIndex outIndex) const {
  if (inIndex == getRemoteBufferOffsetInIndex() &&
      output->hasIndex(getCachedTensorOutIndex())) {
    auto emptyRegion =
        view::Region::getEmpty(outRank(getCachedTensorOutIndex()));
    return [emptyRegion](const view::Region &) {
      return view::Regions(1, emptyRegion);
    };
  }
  return Op::fwdRegMap(inIndex, outIndex);
}

view::RegMap CacheLoadOp::bwdRegMap(InIndex inIndex, OutIndex outIndex) const {
  if (inIndex == getRemoteBufferOffsetInIndex() &&
      output->hasIndex(getCachedTensorOutIndex())) {
    auto emptyRegion =
        view::Region::getEmpty(inRank(getRemoteBufferOffsetInIndex()));
    return [emptyRegion](const view::Region &) {
      return view::Regions(1, emptyRegion);
    };
  }
  return Op::bwdRegMap(inIndex, outIndex);
}

static OpDefinition::DataTypes T = {DataType::FLOAT,
                                    DataType::FLOAT16,
                                    DataType::INT32,
                                    DataType::UINT32};

static OpDefinition
    cacheLoadOpDef({OpDefinition::Inputs({{"X", T}, {"O", {DataType::INT32}}}),
                    OpDefinition::Outputs({{"Y", T}}),
                    OpDefinition::Attributes({})});

static OpDefinition
    cacheStoreOpDef({OpDefinition::Inputs({{"X", T}, {"O", {DataType::INT32}}}),
                     OpDefinition::Outputs({}),
                     OpDefinition::Attributes({})});

static OpCreator<CacheLoadOp> cacheLoadOpCreator(
    OpDefinitions({{Onnx::CustomOperators::CacheLoad, cacheLoadOpDef}}),
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr = {}) -> std::unique_ptr<Op> {
      int64_t bufferid = attr.getAttribute<Attributes::Int>("bufferid");
      return std::unique_ptr<CacheLoadOp>(
          new CacheLoadOp(_opid, settings, bufferid));
    },
    true);

static OpCreator<CacheLoadOp> cacheStoreOpCreator(
    OpDefinitions({{Onnx::CustomOperators::CacheStore, cacheStoreOpDef}}),
    [](const OperatorIdentifier &_opid,
       const Op::Settings &settings,
       const Attributes &attr = {}) -> std::unique_ptr<Op> {
      int64_t bufferid = attr.getAttribute<Attributes::Int>("bufferid");
      return std::unique_ptr<CacheStoreOp>(
          new CacheStoreOp(_opid, settings, bufferid));
    },
    true);

} // namespace popart
