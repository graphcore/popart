// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <algorithm>
#include <onnx/onnx_pb.h>
#include <vector>
#include <popart/ces/slicece.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/slice.hpp>
#include <popart/tensor.hpp>

namespace popart {

class IndicesIter {
public:
  IndicesIter(const std::vector<Slice> &slices_) : slices(slices_) {
    // initialize indices
    indices.reserve(slices.size());
    for (auto slice : slices_) {
      indices.push_back(slice.start);
    }

    // check slices are contiguous
    for (int i = 0; i < slices.size(); i++) {
      if (slices[i].axis != i) {
        throw error("slices must be contiguous");
      }
    }
  }

  const std::vector<int64_t> &operator*() const { return indices; }

  // Advances indices by incrementing fastest index,
  // carrying overflows to slower indices.
  void operator++(int) {
    for (int64_t i = indices.size() - 1; i >= 0; i--) {
      indices[i] += 1;
      if (indices[i] < slices[i].end) {
        return;
      } else {
        indices[i] = slices[i].start;
      }
    }
  }

private:
  const std::vector<Slice> slices;
  std::vector<int64_t> indices;
};

ConstExprSlice::ConstExprSlice(Op *op_) : ConstExprOp(op_) {}

std::vector<Slice> ConstExprSlice::getAllSlices() {
  auto in_info = inInfo(0);
  std::vector<Slice> slices;
  slices.reserve(in_info.rank());

  // create default slices
  for (int i = 0; i < in_info.rank(); i++) {
    slices.emplace_back(0, in_info.dim(i), i);
  }

  // if there is a slice for an axis in slice_op
  // replace the default slice with the slice_op slice
  for (auto slice : getOp<BaseSliceOp>().getSlices()) {
    slices[slice.axis] = slice;
  }

  return slices;
}

class SliceFunctor {
public:
  template <typename T>
  std::vector<char> operator()(Tensor &input,
                               const TensorInfo &outInfo,
                               const std::vector<Slice> &slices) {
    std::vector<char> v_out(outInfo.nbytes());
    T *output = reinterpret_cast<T *>(v_out.data());
    NDArrayWrapper<T> data0(input);

    auto indices = IndicesIter(slices);
    for (int64_t i = 0; i < outInfo.nelms(); i++) {
      output[i] = data0[*indices];
      indices++;
    }

    return v_out;
  }
};

std::vector<char> ConstExprSlice::compute() {
  Tensor *in0 = inTensor(0);

  return callOpFunctor<SliceFunctor>(
      in0->info.dataType(), *in0, outInfo0(), getAllSlices());
}

} // namespace popart
