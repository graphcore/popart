#include <poponnx/ces/castce.hpp>
#include <poponnx/onnxutil.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

void CastCe::insertOutput() {

  // Obtain the output type
  int64_t i64_to;
  nAtts.set(i64_to, "to");
  auto tpdt_to   = static_cast<onnx::TensorProto_DataType>(i64_to);
  DataType dt_to = onnxutil::getDataType(tpdt_to);
  Tensor *in0    = atInIndex(0);
  TensorInfo outInfo{dt_to, in0->info.shape()};

  // initialize a container for the output data
  std::vector<char> v_out(outInfo.nbytes());

  // We handle one case here as a proof of concept : INT32 -> FLOAT.
  // To scale this to all possible pairs, we will need a different approach.
  // see T5925
  if (in0->info.dataType() == DataType::INT32 && dt_to == DataType::FLOAT) {
    auto input  = static_cast<int *>(in0->tensorData()->data());
    auto output = reinterpret_cast<float *>(v_out.data());
    for (int i = 0; i < outInfo.nelms(); ++i) {
      output[i] = static_cast<float>(input[i]);
    }
  }

  else {
    throw error("Currently no support for casting from " +
                in0->info.data_type() + " to " + outInfo.data_type());
  }
  addConstInitTensor(atOutIndex0(), in0->info, v_out.data());
}

} // namespace poponnx
