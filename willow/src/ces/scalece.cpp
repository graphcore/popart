#include <poponnx/ces/scalece.hpp>
#include <poponnx/onnxutil.hpp>
#include <poponnx/op/scale.hpp>
#include <poponnx/tensor.hpp>

namespace poponnx {

// custom_cast is just a static_cast unless a specialization is provided
template <typename To> To custom_cast(double x) { return static_cast<To>(x); }

// specialize custom_cast in the cast of double to Half to prevent compiler
// warning `implicit conversion 'double' to 'float'`
template <> Half custom_cast(double x) {
  return static_cast<Half>(static_cast<float>(x));
}

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
      T outval  = custom_cast<T>(static_cast<double>(inval) * factor64);
      output[i] = outval;
    }
    return v_out;
  }
};

ConstExprScale::ConstExprScale(Op *op_) : ConstExprOp(op_) {
  factor32 = getOp<ScaleOp>().getScaleFactor();
  factor64 = static_cast<double>(factor32);
}

std::vector<char> ConstExprScale::compute() {

  // The tensor which will be scaled
  Tensor *in0 = inTensor(0);

  auto data = callOpFunctor<ScaleFunctor>(
      in0->info.dataType(), // callOpFunctor will determine what template
                            // parameter to use from this poponnx type
      in0,                  // arg0 of ScaleFunctor
      factor64              // arg1 of scaleFunctor
  );

  return data;
}

} // namespace poponnx
