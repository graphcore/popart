// Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ext/new_allocator.h>
#include <memory>
#include <queue>
#include <vector>
#include <poplar/Tensor.hpp>
#include <popops/ElementWise.hpp>
#include <popops/Expr.hpp>
#include <popops/ExprOp.hpp>
#include <popops/OperationDef.hpp>
#include <popops/Reduce.hpp>
#include <popart/op/mean.hpp>
#include <popart/popx/op/meanx.hpp>
#include <popart/popx/opxmanager.hpp>
#include <popart/tensorindex.hpp>

#include "popart/op.hpp"
#include "popart/operatoridentifier.hpp"
#include "popart/operators.hpp"
#include "popart/popx/op/elementwisex.hpp"
#include "popart/popx/opx.hpp"
#include "popart/tensorinfo.hpp"
#include "popart/util.hpp"

namespace poplar {
namespace program {
class Sequence;
} // namespace program
} // namespace poplar

namespace pe = popops::expr;

namespace popart {
namespace popx {
class Devicex;

MeanOpx::MeanOpx(Op *op, Devicex *devicex) : ElementWiseUnaryOpx(op, devicex) {
  verifyOp<MeanOp>(op, {Onnx::Operators::Mean_8, Onnx::Operators::Mean_6});
}

void MeanOpx::grow(poplar::program::Sequence &prog) const {
  auto outTensor = cloneNcopy(prog, getInTensor(0));

  if (op_p->input->n() > 1) {
    // Follow the logic in the sumx op to build the sum operation over
    // the input tensors.
    // Holder for the input tensors:
    std::vector<poplar::Tensor> inputs;

    // The "owner" of all expr nodes:
    std::vector<std::unique_ptr<popops::expr::Expr>> exprs;

    // The queue of expr nodes to be reduced:
    std::queue<popops::expr::Expr *> expr;

    for (int i = 0; i < op_p->input->n(); ++i) {
      inputs.push_back(getInTensor(i));
      exprs.push_back(std::make_unique<pe::PlaceHolder>(i + 1));
      expr.push(exprs.back().get());
    }

    // Build a fairly balanced binary tree
    while (expr.size() > 1) {
      auto &a = *expr.front();
      expr.pop();
      auto &b = *expr.front();
      expr.pop();

      exprs.push_back(std::make_unique<pe::Add>(a, b));
      expr.push(exprs.back().get());
    }
    // Add in a divide in the end.
    outTensor =
        popops::map(graph(),
                    pe::Divide(*expr.front(), pe::Const(op_p->input->n())),
                    inputs,
                    prog,
                    debugContext("mean"));
  }

  setOutTensor(MeanOp::getOutIndex(), outTensor);
}

MeanArgGradOpx::MeanArgGradOpx(Op *op_, Devicex *devicex_)
    : Opx(op_, devicex_) {}

void MeanArgGradOpx::grow(poplar::program::Sequence &prog) const {
  auto &gradOp = getOp<MeanArgGradOp>();

  auto shapeOfInputToBwdOp = inInfo(MeanArgGradOp::getGradInIndex()).shape();
  auto shapeOfInputToFwdOp = gradOp.getFwdInputInfo().shape();

  // Create the axes to reduce along.
  std::vector<int64_t> axes =
      npReductionAxis(shapeOfInputToFwdOp, shapeOfInputToBwdOp);

  // Remove axes from the result that were not present ( or 1) in the input to
  // the fwd op
  auto out = popops::reduce(graph(),
                            getInTensor(MeanArgGradOp::getGradInIndex()),
                            vXtoY<int64_t, std::size_t>(axes),
                            {popops::Operation::ADD},
                            prog,
                            debugContext("reduce"));

  // scale the grad input (reduced)
  popops::mapInPlace(graph(),
                     pe::Mul(pe::_1, pe::Const(gradOp.getScale())),
                     {out},
                     prog,
                     debugContext("mull"));

  // Reshape the output, to add 1's if needed
  setOutTensor(MeanArgGradOp::getOutIndex(),
               out.reshape(outInfo(MeanArgGradOp::getOutIndex()).shape_szt()));
}

namespace {
OpxCreator<MeanOpx> meanOpxCreator({Onnx::Operators::Mean_6,
                                    Onnx::Operators::Mean_8});
OpxCreator<MeanArgGradOpx> meanGradOpxCreator(Onnx::GradOperators::MeanArgGrad);
} // namespace

} // namespace popx
} // namespace popart
