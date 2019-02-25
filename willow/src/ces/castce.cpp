#include <poponnx/ces/castce.hpp>
#include <poponnx/onnxutil.hpp>
#include <poponnx/op/cast.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

ConstExprCast::ConstExprCast(Op *op_) : ConstExprOp(op_) {}

std::vector<char> ConstExprCast::compute() {
  // Obtain the output type
  Tensor *in0          = inTensor(0);
  const auto &out_info = outInfo0();
  DataType dt_to       = out_info.dataType();

  // initialize a container for the output data
  std::vector<char> v_out(out_info.nbytes());

  if (in0->info.dataType() == dt_to) {
    std::memcpy(v_out.data(), in0->tensorData()->data(), in0->info.nbytes());
  }

  // We handle one  interesting case as a proof of concept : INT32 -> FLOAT.
  // To scale this to all possible pairs, we will need a different approach.
  // see T5925
  //
  else if (in0->info.dataType() == DataType::INT32 &&
           dt_to == DataType::FLOAT) {
    auto input  = static_cast<int *>(in0->tensorData()->data());
    auto output = reinterpret_cast<float *>(v_out.data());
    for (int i = 0; i < out_info.nelms(); ++i) {
      output[i] = static_cast<float>(input[i]);
    }
  } else if (in0->info.dataType() == DataType::INT32 &&
             dt_to == DataType::FLOAT16) {
    auto input  = static_cast<int *>(in0->tensorData()->data());
    auto output = reinterpret_cast<float16_t *>(v_out.data());
    for (int i = 0; i < out_info.nelms(); ++i) {
      output[i] = static_cast<float16_t>(input[i]);
    }
  } else if (in0->info.dataType() == DataType::FLOAT16 &&
             dt_to == DataType::FLOAT) {
    auto input  = static_cast<float16_t *>(in0->tensorData()->data());
    auto output = reinterpret_cast<float_t *>(v_out.data());
    for (int i = 0; i < out_info.nelms(); ++i) {
      output[i] = static_cast<float_t>(input[i]);
    }
  }

  else {
    throw error("Currently no support for casting from " +
                in0->info.data_type() + " to " + out_info.data_type());
  }

  return v_out;
}

} // namespace poponnx
