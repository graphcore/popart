#include <poponnx/error.hpp>
#include <poponnx/makeunique.hpp>
#include <poponnx/op.hpp>
#include <poponnx/popx/devicex.hpp>
#include <poponnx/popx/op/poolx.hpp>

#include <popnn/Pooling.hpp>

namespace poponnx {
namespace popx {

PoolOpx::PoolOpx(Op *op, Devicex *device) : Opx(op, device) {}

static poplar::Type getReductionType(const popnn::PoolingType &pooling_type,
                                     const poplar::Type &input_type) {
  switch (pooling_type) {
  case popnn::PoolingType::AVG:
  case popnn::PoolingType::SUM:
    return poplar::FLOAT;
  case popnn::PoolingType::MAX:
    return input_type;
  }
}

popnn::pooling::PoolParams
PoolOpx::GetPoolingParameters(const popnn::PoolingType &pooling_type,
                              const TensorInfo &input_tensor,
                              const std::vector<int64_t> &kernel,
                              const std::vector<int64_t> &strides,
                              const std::vector<int64_t> &lowerPads,
                              const std::vector<int64_t> &upperPads) const {

  const auto &input_shape = input_tensor.shape_szt();

  const auto batch_size   = input_shape[0];
  const auto num_channels = input_shape[1];
  std::vector<std::size_t> input_field_shape(std::next(input_shape.begin(), 2),
                                             input_shape.end());

  std::vector<std::size_t> field_shape;
  std::vector<std::size_t> kernel_shape;
  std::vector<unsigned> stride;
  std::vector<int> padding_lower;
  std::vector<int> padding_upper;

  for (int d = 0; d < kernel.size(); d++) {
    field_shape.push_back(input_field_shape[d]);
    kernel_shape.push_back(kernel[d]);
    stride.push_back(static_cast<unsigned int>(strides[d]));
    padding_lower.push_back(static_cast<int>(lowerPads[d]));
    padding_upper.push_back(static_cast<int>(upperPads[d]));
  }

  auto data_type = getReductionType(pooling_type, popType(input_tensor));

  return {pooling_type,
          field_shape,
          kernel_shape,
          stride,
          padding_lower,
          padding_upper,
          num_channels,
          batch_size,
          data_type};
}

} // namespace popx
} // namespace poponnx
