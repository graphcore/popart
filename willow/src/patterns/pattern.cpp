#include <poponnx/error.hpp>
#include <poponnx/ir.hpp>
#include <poponnx/patterns/pattern.hpp>
#include <poponnx/tensor.hpp>
#include <poponnx/util.hpp>

namespace poponnx {

bool Pattern::touchesAnchored(Op *op) const {
  for (auto &tensor : touches(op)) {
    if (op->pir->isAnchored(tensor->id)) {
      return true;
    }
  }
  return false;
};

} // namespace poponnx
