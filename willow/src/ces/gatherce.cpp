// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <vector>
#include <popart/ces/gatherce.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/gather.hpp>
#include <popart/tensor.hpp>

namespace popart {

ConstExprGather::ConstExprGather(Op *op_) : ConstExprOp(op_) {}

class GatherFunctor {
public:
  template <typename T>
  std::vector<char> operator()(Tensor &dataIn,
                               Tensor &indicesIn,
                               int64_t axis,
                               const TensorInfo &outInfo) {

    auto inShape = dataIn.info.shape();
    std::vector<char> v_out(outInfo.nbytes());
    T *output = reinterpret_cast<T *>(v_out.data());
    NDArrayWrapper<T> data0(dataIn);
    NDArrayWrapper<int64_t> data1(indicesIn);

    const int64_t axis_size     = inShape[axis];
    const int64_t indices_count = indicesIn.info.nelms();

    // Size of the array below the gather axis
    int64_t outer_size = 1;
    for (int64_t i = 0; i < axis; ++i) {
      outer_size *= inShape[i];
    }

    // Size of the array beyond the gather axis
    int64_t inner_size = 1;
    for (int64_t i = axis + 1; i < inShape.size(); ++i) {
      inner_size *= inShape[i];
    }

    // Outer elements
    for (int64_t outer = 0; outer < outer_size; ++outer) {
      // Specified indices to copy
      for (int64_t i = 0; i < indices_count; ++i) {
        // All inner elements can be copied contiguously
        for (int64_t j = 0; j < inner_size; ++j) {
          output[(outer * indices_count + i) * inner_size] =
              data0[(outer * axis_size + data1[i]) * inner_size];
        }
      }
    }

    return v_out;
  }
};

std::vector<char> ConstExprGather::compute() {
  Tensor *dataIn    = inTensor(GatherOp::dataInIndex());
  Tensor *indicesIn = inTensor(GatherOp::indicesInIndex());
  int64_t axis      = getOp<GatherOp>().getAxis();
  return callOpFunctor<GatherFunctor>(
      dataIn->info.dataType(), *dataIn, *indicesIn, axis, outInfo0());
}

} // namespace popart
