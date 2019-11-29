#include <popart/ces/castce.hpp>
#include <popart/onnxutil.hpp>
#include <popart/op/cast.hpp>
#include <popart/tensor.hpp>

namespace popart {

ConstExprCast::ConstExprCast(Op *op_) : ConstExprOp(op_) {}

// If a specialised conversion is required, a specialised template for doCast
// can be implemented.
template <typename FROM, typename TO>
std::vector<char> doCast(Tensor *inputTensor, const TensorInfo &outputInfo) {
  auto inputData = static_cast<FROM *>(inputTensor->tensorData()->data());

  std::vector<char> output(outputInfo.nbytes());
  auto outputData = reinterpret_cast<TO *>(output.data());

  for (int i = 0; i < outputInfo.nelms(); i++) {
    outputData[i] = static_cast<TO>(inputData[i]);
  }

  return output;
}

template <typename FROM>
std::vector<char> tryCastFrom(Tensor *inputTensor,
                              const TensorInfo &outputInfo) {
  switch (outputInfo.dataType()) {
  case DataType::INT32:
    return doCast<FROM, int32_t>(inputTensor, outputInfo);
  case DataType::INT64:
    return doCast<FROM, int64_t>(inputTensor, outputInfo);
  case DataType::FLOAT:
    return doCast<FROM, float>(inputTensor, outputInfo);
  case DataType::FLOAT16:
    return doCast<FROM, float16_t>(inputTensor, outputInfo);
  case DataType::UINT32:
    return doCast<FROM, uint32_t>(inputTensor, outputInfo);
  case DataType::UINT8:
  case DataType::INT8:
  case DataType::UINT16:
  case DataType::INT16:
  case DataType::UINT64:
  case DataType::BOOL:
  case DataType::BFLOAT16:
  case DataType::DOUBLE:
  case DataType::COMPLEX64:
  case DataType::COMPLEX128:
  case DataType::STRING:
  case DataType::UNDEFINED:
  default:
    throw error("Currently no support for casting from {} to {}",
                inputTensor->info.data_type(),
                outputInfo.data_type());
  }
}

namespace {

std::vector<char> tryCast(Tensor *inputTensor, const TensorInfo &outputInfo) {
  switch (inputTensor->info.dataType()) {
  case DataType::INT32:
    return tryCastFrom<int32_t>(inputTensor, outputInfo);
  case DataType::INT64:
    return tryCastFrom<int64_t>(inputTensor, outputInfo);
  case DataType::FLOAT:
    return tryCastFrom<float>(inputTensor, outputInfo);
  case DataType::FLOAT16:
    return tryCastFrom<float16_t>(inputTensor, outputInfo);
  case DataType::UINT32:
    return tryCastFrom<uint32_t>(inputTensor, outputInfo);
  case DataType::UINT8:
  case DataType::INT8:
  case DataType::UINT16:
  case DataType::INT16:
  case DataType::UINT64:
  case DataType::BOOL:
  case DataType::BFLOAT16:
  case DataType::DOUBLE:
  case DataType::COMPLEX64:
  case DataType::COMPLEX128:
  case DataType::STRING:
  case DataType::UNDEFINED:
  default:
    throw error("Currently no support for casting from data type {}",
                inputTensor->info.data_type());
  }
}

} // namespace

std::vector<char> ConstExprCast::compute() {
  // Obtain the output type
  Tensor *in0          = inTensor(0);
  const auto &out_info = outInfo0();

  if (in0->info.dataType() == out_info.dataType()) {
    std::vector<char> v_out(out_info.nbytes());
    std::memcpy(v_out.data(), in0->tensorData()->data(), in0->info.nbytes());
    return v_out;
  } else {
    return tryCast(in0, out_info);
  }
}

} // namespace popart
