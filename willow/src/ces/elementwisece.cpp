#include <cmath>
#include <vector>
#include <poponnx/ces/elementwisece.hpp>
#include <poponnx/ndarraywrapper.hpp>
#include <poponnx/op/add.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

template <typename OPERATION> class BinaryFunctor {
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
      output[i] = OPERATION::invoke(data0[indices], data1[indices]);
    }
    return v_out;
  }
};

class Div {
public:
  template <typename T> static T invoke(T &lhs, T &rhs) { return lhs / rhs; }
};

class Add {
public:
  template <typename T> static T invoke(T &lhs, T &rhs) { return lhs + rhs; }
};

class Mul {
public:
  template <typename T> static T invoke(T &lhs, T &rhs) { return lhs * rhs; }
};

class Sub {
public:
  template <typename T> static T invoke(T &lhs, T &rhs) { return lhs - rhs; }
};

class Mod {
public:
  template <typename T> static T invoke(T &lhs, T &rhs) { return lhs % rhs; }
};

// template specializations for float & double which use the
// fmod functions
template <> float Mod::invoke<float>(float &lhs, float &rhs) {
  return fmodf(lhs, rhs);
}
template <> double Mod::invoke<double>(double &lhs, double &rhs) {
  return fmod(lhs, rhs);
}

template <> Half Mod::invoke<Half>(Half &lhs, Half &rhs) {
  return fmodf(lhs, rhs);
}

ConstExprDiv::ConstExprDiv(Op *op_) : ConstExprOp(op_) {}

std::vector<char> ConstExprDiv::compute() {
  Tensor *in0 = inTensor(AddOp::getArg0InIndex());
  Tensor *in1 = inTensor(AddOp::getArg1InIndex());
  return callOpFunctor<BinaryFunctor<Div>>(in0->info.dataType(), *in0, *in1);
}

ConstExprAdd::ConstExprAdd(Op *op_) : ConstExprOp(op_) {}

std::vector<char> ConstExprAdd::compute() {
  Tensor *in0 = inTensor(AddOp::getArg0InIndex());
  Tensor *in1 = inTensor(AddOp::getArg1InIndex());
  return callOpFunctor<BinaryFunctor<Add>>(in0->info.dataType(), *in0, *in1);
}

ConstExprMul::ConstExprMul(Op *op_) : ConstExprOp(op_) {}

std::vector<char> ConstExprMul::compute() {
  Tensor *in0 = inTensor(AddOp::getArg0InIndex());
  Tensor *in1 = inTensor(AddOp::getArg1InIndex());
  return callOpFunctor<BinaryFunctor<Mul>>(in0->info.dataType(), *in0, *in1);
}

ConstExprSub::ConstExprSub(Op *op_) : ConstExprOp(op_) {}

std::vector<char> ConstExprSub::compute() {
  Tensor *in0 = inTensor(AddOp::getArg0InIndex());
  Tensor *in1 = inTensor(AddOp::getArg1InIndex());
  return callOpFunctor<BinaryFunctor<Sub>>(in0->info.dataType(), *in0, *in1);
}

ConstExprMod::ConstExprMod(Op *op_) : ConstExprOp(op_) {}

std::vector<char> ConstExprMod::compute() {
  Tensor *in0 = inTensor(AddOp::getArg0InIndex());
  Tensor *in1 = inTensor(AddOp::getArg1InIndex());
  return callOpFunctor<BinaryFunctor<Mod>>(in0->info.dataType(), *in0, *in1);
}

} // namespace poponnx
