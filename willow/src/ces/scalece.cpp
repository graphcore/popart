#include <poponnx/ces/scalece.hpp>
#include <poponnx/onnxutil.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

class ScaleFunctor {
public:
  template <typename T>
  std::vector<char> operator()(Tensor *in0, double factor64) {

    TensorInfo outInfo = in0->info;
    // initialize a container for the output data
    std::vector<char> v_out(outInfo.nbytes());

    auto input  = static_cast<T *>(in0->tensorData()->data());
    auto output = reinterpret_cast<T *>(v_out.data());
    for (int i = 0; i < outInfo.nelms(); ++i) {
      T inval = input[i];
      // cast all to double before scaling, then return to type T
      T outval  = static_cast<T>(static_cast<double>(inval) * factor64);
      output[i] = outval;
    }
    return v_out;
  }
};

template <>
std::vector<char> ScaleFunctor::operator()<poponnx::Half>(Tensor *, double) {
  throw error("cannot const expr scale half tensor for now");
}

ConstExprScale::ConstExprScale(const onnx::NodeProto &n, Ir *i)
    : ConstExprOp(n, i) {
  nAtts.set(factor32, "scale");
  factor64 = static_cast<double>(factor32);
}

void ConstExprScale::insertOutput() {

  // The tensor which will be scaled
  Tensor *in0 = atInIndex(0);

  auto data = callOpFunctor<ScaleFunctor>(
      in0->info.dataType(), // callOpFunctor will determine what template
                            // parameter to use from this this poponnx type
      in0,                  // arg0 of ScaleFunctor
      factor64              // arg1 of scaleFunctor
  );

  addConstInitTensor(atOutIndex0(),
                     in0->info, // the output info is the same as int input info
                     data.data());
}

} // namespace poponnx
