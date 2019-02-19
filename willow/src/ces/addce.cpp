#include <vector>
#include <poponnx/ces/addce.hpp>
#include <poponnx/ndarraywrapper.hpp>
#include <poponnx/op/add.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

ConstExprAdd::ConstExprAdd(Op *op) : ConstExprOp(op) {}

// add two Tensors together using numpy-broadcasting,
// return the data as a vector<char>
class AddFunctor {
public:
  template <typename T> std::vector<char> operator()(Tensor &in0, Tensor &in1) {
    TensorInfo outInfo = npOut(in0.info, in1.info);
    std::vector<char> v_out(outInfo.nbytes());
    NDArrayWrapper<T> output(reinterpret_cast<T *>(v_out.data()), outInfo);
    NDArrayWrapper<T> data0(in0);
    NDArrayWrapper<T> data1(in1);
    for (int64_t i = 0; i < output.nelms(); ++i) {
      // the N-dimensional indices in the output tensor
      auto indices = output.unflatten(i);
      // perform the addition, where the broadcasting of the
      // operands is implicitly taken care of by NDIndices
      output[i] = data0[indices] + data1[indices];
    }
    return v_out;
  }
};

std::vector<char> ConstExprAdd::compute() {
  Tensor *in0 = inTensor(AddOp::getArg0InIndex());
  Tensor *in1 = inTensor(AddOp::getArg1InIndex());
  return callOpFunctor<AddFunctor>(in0->info.dataType(), *in0, *in1);
}

} // namespace poponnx
