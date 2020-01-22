#include <algorithm>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/op/cache.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/tensor.hpp>

namespace popart {

CacheStoreOp::CacheStoreOp(const OperatorIdentifier &_opid,
                           const Op::Settings &settings_)
    : Op(_opid, settings_), remotebuffer_id(-1) {}

std::unique_ptr<Op> CacheStoreOp::clone() const {
  return std::make_unique<CacheStoreOp>(*this);
}

CacheLoadOp::CacheLoadOp(const OperatorIdentifier &_opid,
                         const TensorInfo &tensor_info_,
                         const Op::Settings &settings_)
    : Op(_opid, settings_), tensor_info(tensor_info_), remotebuffer_id(-1) {}

std::unique_ptr<Op> CacheLoadOp::clone() const {
  return std::make_unique<CacheLoadOp>(*this);
}

void CacheLoadOp::setup() { outInfo(getCachedTensorOutIndex()) = tensor_info; }

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

} // namespace popart
