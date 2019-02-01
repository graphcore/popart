#ifndef GUARD_NEURALNET_CONSTEXPRS_SLICECE_HPP
#define GUARD_NEURALNET_CONSTEXPRS_SLICECE_HPP

#include <boost/optional.hpp>
#include <poponnx/ces/constexpr.hpp>
#include <poponnx/op/slice.hpp>

namespace poponnx {

class ConstExprSlice : public ConstExprOp {
public:
  ConstExprSlice(const onnx::NodeProto &n, Ir *i);
  void insertOutput() final;

private:
  static SliceImpl createSliceImpl(const onnx::NodeProto &n);
  std::vector<Slice> getAllSlices();

  SliceImpl impl;
};

} // namespace poponnx

#endif
