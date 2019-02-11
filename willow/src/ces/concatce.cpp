#include <onnx/onnx_pb.h>
#include <poponnx/ces/concatce.hpp>
#include <poponnx/ndarraywrapper.hpp>
#include <poponnx/ndindices.hpp>
#include <poponnx/op/concat.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

ConstExprConcat::ConstExprConcat(const onnx::NodeProto &n, Ir *i)
    : ConstExprOp(n, i) {
  axis        = nAtts.getAttribute<Attributes::Int>("axis");
  input_count = node.input_size();
}

class ConcatFunctor {
public:
  template <typename T>
  std::vector<char> operator()(const TensorInfo &out_info,
                               std::vector<Tensor *> inputs,
                               int64_t axis) {
    std::vector<char> v_out(out_info.nbytes());
    NDArrayWrapper<T> output(reinterpret_cast<T *>(v_out.data()), out_info);

    auto stride = inputs[0]->info.dim(static_cast<int>(axis));

    for (int i = 0; i < inputs.size(); i++) {
      auto input = inputs[i];
      NDArrayWrapper<T> data(*input);
      for (int j = 0; j < input->info.nelms(); j++) {
        auto indices = data.unflatten(j);
        auto pindices(indices);
        pindices[axis] += stride * i;
        output[pindices] = data[indices];
      }
    }

    return v_out;
  }
};

void ConstExprConcat::insertOutput() {
  std::vector<const Shape *> input_shapes;
  input_shapes.reserve(input_count);
  for (int i = 0; i < input_count; i++) {
    input_shapes.push_back(&atInIndex(i)->info.shape());
  }

  auto out_shape = ConcatOp::getOutputShape(axis, input_shapes);
  auto in0_info  = atInIndex(0)->info;
  TensorInfo out_info(in0_info.dataType(), out_shape);

  std::vector<Tensor *> inputs;
  for (int i = 0; i < input_count; i++) {
    inputs.push_back(atInIndex(i));
  }

  auto data =
      callOpFunctor<ConcatFunctor>(in0_info.dataType(), out_info, inputs, axis);
  addConstInitTensor(atOutIndex0(), out_info, data.data());
}

} // namespace poponnx
