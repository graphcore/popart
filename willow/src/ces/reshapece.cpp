#include <string>
#include <vector>

#include <popart/ces/reshapece.hpp>
#include <popart/ndarraywrapper.hpp>
#include <popart/op/reshape.hpp>
#include <popart/tensor.hpp>

namespace popart {

namespace {
class ReshapeFunctor {
public:
  template <typename T>
  std::vector<char> operator()(Tensor &in, const Shape &newShape) {
    TensorInfo outInfo(in.info.data_type(), newShape);
    std::vector<char> v_out(outInfo.nbytes());
    NDArrayWrapper<T> output(reinterpret_cast<T *>(v_out.data()), outInfo);
    NDArrayWrapper<T> data(in);

    std::memcpy(output.data(), data.data(), outInfo.nbytes());
    return v_out;
  }
};
} // namespace

ConstExprReshape::ConstExprReshape(Op *op) : ConstExprOp(op) {}

std::vector<char> ConstExprReshape::compute() {
  std::vector<char> data_;
  Tensor *in0 = inTensor(0);

  auto shape = getOp<ReshapeOp>().getOutShape();

  return callOpFunctor<ReshapeFunctor>(in0->info.dataType(), *in0, shape);
}

} // namespace popart
