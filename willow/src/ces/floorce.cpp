// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <cmath>

#include <popart/ces/floorce.hpp>
#include <popart/onnxutil.hpp>
#include <popart/op/floor.hpp>
#include <popart/tensor.hpp>

namespace popart {

class FloorFunctor {
public:
  template <typename T> std::vector<char> operator()(Tensor *in0) {

    TensorInfo outInfo = in0->info;
    // initialize a container for the output data
    std::vector<char> v_out(outInfo.nbytes());

    auto input  = static_cast<T *>(in0->tensorData()->data());
    auto output = reinterpret_cast<T *>(v_out.data());
    for (int i = 0; i < outInfo.nelms(); ++i) {
      T inval   = input[i];
      T outval  = std::floor(inval);
      output[i] = outval;
    }
    return v_out;
  }
};

ConstExprFloor::ConstExprFloor(Op *op_) : ConstExprOp(op_) {}

std::vector<char> ConstExprFloor::compute() {
  Tensor *in0 = inTensor(0);
  auto data   = callOpFunctor<FloorFunctor>(in0->info.dataType(), in0);
  return data;
}

} // namespace popart
