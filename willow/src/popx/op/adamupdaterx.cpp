// Copyright (c) 2020 Graphcore Ltd. All rights reserved.

#include <cmath>

#include <popops/ElementWise.hpp>
#include <popops/Reduce.hpp>
#include <popops/ScaledAdd.hpp>
#include <popart/error.hpp>
#include <popart/ir.hpp>
#include <popart/op/adamupdater.hpp>
#include <popart/op/varupdate.hpp>
#include <popart/popx/devicex.hpp>
#include <popart/popx/op/adamupdaterx.hpp>
#include <popart/popx/opxmanager.hpp>

namespace pe = popops::expr;

namespace popart {
namespace popx {

AdamUpdaterOpx::AdamUpdaterOpx(Op *op, Devicex *devicex) : Opx(op, devicex) {
  verifyOp<AdamUpdaterOp>(op, Onnx::CustomOperators::AdamUpdater);
}

void AdamUpdaterOpx::grow(poplar::program::Sequence &prog) const {
  auto &adamUpdaterOp = getOp<AdamUpdaterOp>();

  poplar::Tensor var;
  poplar::Tensor accl1  = getInTensor(AdamUpdaterOp::getAccl1InIndex());
  poplar::Tensor accl2  = getInTensor(AdamUpdaterOp::getAccl2InIndex());
  poplar::Type elemType = accl1.elementType();
  std::vector<poplar::Tensor> tensors{accl1, accl2};

  int varIndex  = -1;
  int stepIndex = -1;

  if (hasInput(AdamUpdaterOp::getVarInIndex())) {
    var      = getInTensor(AdamUpdaterOp::getVarInIndex());
    elemType = var.elementType();
    tensors.push_back(var);
    varIndex = tensors.size();
  } else if (!adamUpdaterOp.initWd.isConst() ||
             adamUpdaterOp.initWd.val() > 0.0f) {
    throw error("Expected {} to have the variable tensor connected at index {}",
                adamUpdaterOp.debugName(),
                AdamUpdaterOp::getVarInIndex());
  }

  // Calculate updater term for both const & tensor optimizer parameters
  if (hasInput(AdamUpdaterOp::getStepInIndex())) {
    poplar::Tensor step = getInTensor(AdamUpdaterOp::getStepInIndex());

    // Update step
    popops::mapInPlace(graph(), pe::Add(pe::_1, pe::Const(1)), {step}, prog);

    tensors.push_back(step);
    stepIndex = tensors.size();
  }

  pe::Any expr(pe::Const(0.0f));
  pe::Any b1correction(pe::Const(1.0f));
  pe::Any b2correction(pe::Const(1.0f));
  pe::Any mhat(pe::Const(0.0f));
  pe::Any vhat(pe::Const(0.0f));

  // With bias correction (Adam, Lamb)
  // otherwise without (AdamNoBias, LambNoBias, AdaMax)
  if (adamUpdaterOp.mode == AdamMode::Adam ||
      adamUpdaterOp.mode == AdamMode::Lamb ||
      adamUpdaterOp.mode == AdamMode::AdaMax) {
    // b1correction: (1 - b_1^t)
    if (adamUpdaterOp.initB1.isConst()) {
      b1correction =
          pe::Sub(pe::Const(1.0f),
                  pe::Pow(pe::Const(adamUpdaterOp.initB1.val()),
                          pe::Cast(pe::PlaceHolder(stepIndex), poplar::FLOAT)));
    } else {
      tensors.push_back(getInTensor(AdamUpdaterOp::getBeta1InIndex()));
      b1correction =
          pe::Sub(pe::Const(1.0f),
                  pe::Pow(pe::PlaceHolder(tensors.size()),
                          pe::Cast(pe::PlaceHolder(stepIndex), poplar::FLOAT)));
    }

    // Casting here since it's safe (b1correction and b2correction will have
    // at least one non-const component)
    b1correction = pe::Cast(b1correction, accl1.elementType());

    if (adamUpdaterOp.mode != AdamMode::AdaMax) {
      // b2correction: (1 - b_2^t)
      if (adamUpdaterOp.initB2.isConst()) {
        b2correction = pe::Sub(
            pe::Const(1.0f),
            pe::Pow(pe::Const(adamUpdaterOp.initB2.val()),
                    pe::Cast(pe::PlaceHolder(stepIndex), poplar::FLOAT)));
      } else {
        tensors.push_back(getInTensor(AdamUpdaterOp::getBeta2InIndex()));
        b2correction = pe::Sub(
            pe::Const(1.0f),
            pe::Pow(pe::PlaceHolder(tensors.size()),
                    pe::Cast(pe::PlaceHolder(stepIndex), poplar::FLOAT)));
      }

      // Casting here since it's safe (b1correction and b2correction will have
      // at least one non-const component)
      b2correction = pe::Cast(b2correction, accl2.elementType());
    }
  }

  // Accl1 (m) -> mhat
  mhat = pe::Divide(pe::PlaceHolder(1), b1correction);

  // Accl2 (v) -> vhat
  vhat = pe::Divide(pe::PlaceHolder(2), b2correction);

  if (adamUpdaterOp.mode != AdamMode::AdaMax) {
    vhat = pe::Sqrt(vhat);
  }

  // Update term (without weight decay) ->
  //   mhat/(sqrt(vhat) + eps)    (Adam/Lamb)
  //   mhat/(vhat + eps)          (AdaMax)
  if (adamUpdaterOp.initEps.isConst()) {
    expr = pe::Divide(pe::Cast(mhat, accl2.elementType()),
                      pe::Add(vhat, pe::Const(adamUpdaterOp.initEps.val())));
  } else {
    tensors.push_back(getInTensor(AdamUpdaterOp::getEpsInIndex()));
    expr = pe::Divide(pe::Cast(mhat, accl2.elementType()),
                      pe::Add(vhat,
                              pe::Cast(pe::PlaceHolder(tensors.size()),
                                       accl2.elementType())));
  }

  // AdamW (weight decay)
  if (adamUpdaterOp.initWd.isConst()) {
    if (adamUpdaterOp.initWd.val() == 0.0f) {
      // No weight decay, expr stays unchanged
    } else {
      // Constant weight decay
      expr = pe::Add(pe::Cast(expr, elemType),
                     pe::Mul(pe::Const(adamUpdaterOp.initWd.val()),
                             pe::PlaceHolder(varIndex)));
    }
  } else {
    // Non-const weight decay
    tensors.push_back(getInTensor(AdamUpdaterOp::getWdInIndex()));
    expr = pe::Add(pe::Cast(expr, elemType),
                   pe::Mul(pe::Cast(pe::PlaceHolder(tensors.size()), elemType),
                           pe::PlaceHolder(varIndex)));
  }

  poplar::Tensor updater = popops::map(
      graph(), pe::Cast(expr, elemType), tensors, prog, debugContext(""));

  if (hasInput(AdamUpdaterOp::getVarInIndex())) {
    if (hasInViewChangers(AdamUpdaterOp::getVarInIndex())) {
      setOutViewChangers(AdamUpdaterOp::getUpdaterOutIndex(),
                         getInViewChangers(AdamUpdaterOp::getVarInIndex()));
    }
  } else {
    if (hasInViewChangers(AdamUpdaterOp::getAccl1InIndex())) {
      setOutViewChangers(AdamUpdaterOp::getUpdaterOutIndex(),
                         getInViewChangers(AdamUpdaterOp::getAccl1InIndex()));
    }
  }

  setOutTensor(AdamUpdaterOp::getUpdaterOutIndex(), updater);
}

namespace {
OpxCreator<AdamUpdaterOpx>
    AdamUpdaterOpxCreator(Onnx::CustomOperators::AdamUpdater);
}
} // namespace popx
} // namespace popart
