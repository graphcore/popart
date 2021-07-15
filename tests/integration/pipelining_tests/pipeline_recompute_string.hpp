// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <array>
#include <sstream>
#include <popart/ir.hpp>
#include <popart/op.hpp>
#include <popart/op/ipucopy.hpp>

namespace pipeline_recompute_util {

using namespace popart;
template <size_t nIpus>
void fillLogStreams(std::array<std::stringstream, nIpus> &sss,
                    const std::vector<Op *> &sched) {

  auto recStr = [](RecomputeType t) {
    return t == RecomputeType::Checkpoint ? "ChPo" : "Reco";
  };

  for (auto op : sched) {
    auto ipuCopyOp = dynamic_cast<IpuCopyOp *>(op);
    if (!ipuCopyOp) {
      auto vgid = op->getVirtualGraphId();
      sss[vgid] << recStr(op->settings.recomputeType) << "  "
                << op->wrtLossStr() << "  " << op->getName() << "     "
                << op->str() << "\n";

      // op->append(sss[vgid]);

    } else {
      int64_t vgid = 0;
      if (ipuCopyOp->getMinSourceIpu() == ipuCopyOp->getMaxSourceIpu()) {
        vgid = ipuCopyOp->getSourceIpu();
      } else {
        // nMultiSource += 1;
      }

      sss[vgid] << recStr(op->settings.recomputeType) << "  "
                << op->wrtLossStr() << "  " << op->getName() << "     "
                << op->str() << "      " << ipuCopyOp->getFromToStr() << "\n";

      // op->append(sss[vgid]);
    }
  }
}
} // namespace pipeline_recompute_util
