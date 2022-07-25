// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <transforms/autodiff/gradgrowersumop.hpp>
#include <tuple>
#include <typeinfo>
#include <utility>
#include <vector>
#include <poprithms/schedule/transitiveclosure/transitiveclosure.hpp>
#include <poprithmstransitiveclosure.hpp>
#include <popart/alias/aliasmodel.hpp>
#include <popart/alias/aliasmodelgrower.hpp>
#include <popart/graph.hpp>
#include <popart/ir.hpp>
#include <popart/names.hpp>
#include <popart/op/add.hpp>
#include <popart/op/init.hpp>
#include <popart/op/sum.hpp>
#include <popart/opmanager.hpp>
#include <popart/transforms/decomposegradsum.hpp>

#include "popart/error.hpp"
#include "popart/graphcoreoperators.hpp"
#include "popart/logging.hpp"
#include "popart/op.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/tensor.hpp"
#include "popart/tensordebuginfo.hpp"
#include "popart/tensorindex.hpp"
#include "popart/tensornames.hpp"
#include "popart/transforms/transform.hpp"
#include "popart/vertex.hpp"
#include "transforms/autodiff/autodiffhelper.hpp"

namespace popart {

std::size_t DecomposeGradSum::id() {
  return typeid(DecomposeGradSum).hash_code();
}

void DecomposeGradSum::applyAddOpAttributes(Op *op) const {
  op->toLoss   = PathToLoss::No;
  op->fromLoss = PathFromLoss::Yes;
}

std::vector<Op *>
DecomposeGradSum::getDecomposableSumOps(const Graph &graph) const {
  std::vector<Op *> decomposableGradSumOps;
  // An op in the graph is deemed a decomposable GradSumOp if:
  // 1. it is a SumOp
  // 2. its name contains GradGrowerSumOp::getGradSumOpNamePrefix()
  // 3. it produces a tensor with an id that contains reservedGradientPrefix()
  // 4. it has a path from the loss
  // 5. it consumes >2 ActGrad tensors
  for (auto &id_op : graph.getOps()) {
    Op *op = id_op.second.get();
    // 1.
    if (op->isConvertibleTo<SumOp>()) {
      // 2.
      if (op->settings.name.find(GradGrowerSumOp::getGradSumOpNamePrefix()) !=
          std::string::npos) {
        // 3.
        if (op->outId(SumOp::getOutIndex()).find(reservedGradientPrefix()) !=
            std::string::npos) {
          // 4.
          if (op->outTensor(SumOp::getOutIndex())->fromLoss ==
              PathFromLoss::Yes) {
            auto inputs               = op->input->tensors();
            bool allInputsAreActGrads = true;
            for (Tensor *t : inputs) {
              if (t->tensorType() != TensorType::ActGrad) {
                allInputsAreActGrads = false;
              }
            }
            // 5.
            if (inputs.size() > 2 && allInputsAreActGrads) {
              decomposableGradSumOps.push_back(op);
            }
          }
        }
      }
    }
  }

  return decomposableGradSumOps;
}

namespace {
bool init = Transform::registerTransform(new DecomposeGradSum());
}

} // namespace popart
