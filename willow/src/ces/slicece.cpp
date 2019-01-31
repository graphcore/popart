#include <algorithm>
#include <onnx/onnx_pb.h>
#include <vector>
#include <poponnx/ces/slicece.hpp>
#include <poponnx/ndindices.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

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

template <typename T> class NDArray {
public:
  NDArray(T *d, const TensorInfo &i) : data(d), info(i), ndindices(info) {}
  T &at(int64_t i) { return data[i]; }
  T &at(const std::vector<int64_t> &indices) {
    return at(ndindices.flatten(indices));
  }
  T *data;
  const TensorInfo &info;
  NDIndices ndindices;
};

ConstExprSlice::ConstExprSlice(const onnx::NodeProto &n, Ir *i)
    : ConstExprOp(n, i), impl(createSliceImpl(n)) {}

SliceImpl ConstExprSlice::createSliceImpl(const onnx::NodeProto &n) {
  auto atts = Attributes(n.attribute());

  auto starts = atts.getAttribute<Attributes::Ints>("starts", {});
  auto ends   = atts.getAttribute<Attributes::Ints>("ends", {});
  auto axes   = atts.getAttribute<Attributes::Ints>("axes", {});

  return SliceImpl(starts, ends, axes);
}

std::vector<Slice> ConstExprSlice::getAllSlices() {
  auto in_info = atInIndex(0)->info;
  std::vector<Slice> slices;
  slices.reserve(in_info.rank());

  // create default slices
  for (int i = 0; i < in_info.rank(); i++) {
    slices.emplace_back(0, in_info.dim(i), i);
  }

  // if there is a slice for an axis in impl
  // replace the default slice with the impl slice
  for (auto slice : impl.getSlices(in_info.shape())) {
    slices[slice.axis] = slice;
  }

  return slices;
}

template <typename T> std::vector<char> ConstExprSlice::slice() {
  auto input   = atInIndex(0);
  auto outInfo = impl.createOutShape(atInIndex(0)->info);

  std::vector<char> v_out(outInfo.nbytes());
  T *output = reinterpret_cast<T *>(v_out.data());
  NDArray<T> data0(reinterpret_cast<T *>(input->tensorData()->data()),
                   input->info);

  auto slices = getAllSlices();

  auto indices = IndicesIter(slices);
  for (int64_t i = 0; i < outInfo.nelms(); i++) {
    output[i] = data0.at(*indices);
    indices++;
  }

  return v_out;
}

void ConstExprSlice::insertOutput() {
  auto outInfo = impl.createOutShape(atInIndex(0)->info);

  std::vector<char> data_;
  Tensor *in0 = atInIndex(0);

  if (in0->info.dataType() == DataType::FLOAT) {
    data_ = slice<float>();
  } else {
    throw error("Currently ConstExprSlice does not support type {}",
                in0->info.data_type());
  }

  addConstInitTensor(atOutIndex0(), outInfo, data_.data());
}

} // namespace poponnx
