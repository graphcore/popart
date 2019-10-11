#include <limits>
#include <memory>
#include <popart/ir.hpp>
#include <popart/logging.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/opmanager.hpp>
#include <popart/opserialiser.hpp>
#include <popart/region.hpp>
#include <popart/tensornames.hpp>

namespace popart {
VarUpdateOp::VarUpdateOp(const OperatorIdentifier &_opid,
                         const TensorId &varId_,
                         const Op::Settings &settings_)
    : Op(_opid, settings_), varId(varId_), varGradId(getGradId(varId)) {

  // very high priority, so that performed as early as possible
  priority = std::numeric_limits<double>::max();
}

void VarUpdateOp::setup() {
  auto info0 = inInfo(getVarToUpdateInIndex());
  auto info1 = inInfo(getUpdaterInIndex());
  if (info0 != info1) {
    std::ostringstream oss;
    oss << "In VarUpdateOp::setup(), the VarToUpdate has TensorInfo \n"
        << info0 << "\nbut the Updater has TensorInfo\n"
        << info1;

    // TODO T12001 : sort this out (serialize matmuls meets grad accl)
    logging::ir::warn(
        "The options enableVirtualGraphs is deprecated and will be removed in "
        "a future release. Please use virtualGraphMode instead");
  }
  outInfo(getUpdatedVarOutIndex()) = info0;
}

view::Region VarUpdateOp::aliases(InIndex index) const {
  if (index == getVarToUpdateInIndex()) {
    return view::Region::getFull(inShape(index));
  } else {
    return view::Region::getEmpty(inRank(index));
  }
}

view::Region VarUpdateOp::modifies(InIndex index) const {
  return aliases(index);
}

float VarUpdateOp::getSubgraphValue() const {
  // If we have replicated graphs then outline VaruUdates, if possible
  // The motivation for this is the (code) cost of inter-IPU copies, hmm
  if (getIr().getSessionOptions().enableReplicatedGraphs) {
    return getHighSubgraphValue();
  } else {
    return getLowSubgraphValue();
  }
}

namespace {} // namespace

} // namespace popart
