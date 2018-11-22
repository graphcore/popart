#ifndef GUARD_BUILDER_IMPL_H
#define GUARD_BUILDER_IMPL_H

#include <poponnx/builder.hpp>
#include <poponnx/names.hpp>

#include <string>

namespace willow {

/**
 * An implementation of a Builder
 */
class BuilderImpl {
public:
  BuilderImpl();

  TensorId addInputTensor(const TensorInfo &tensorInfo);
  TensorId addInitializedInputTensor(const ConstVoidData &initData);

  void addOutputTensor(const TensorId &arg0);

  // Operations requiring only tensor inputs
  TensorId abs(const std::vector<TensorId> &args);
  TensorId acos(const std::vector<TensorId> &args);
  TensorId acosh(const std::vector<TensorId> &args);
  TensorId add(const std::vector<TensorId> &args);
  TensorId logical_and(const std::vector<TensorId> &args);
  TensorId asin(const std::vector<TensorId> &args);
  TensorId asinh(const std::vector<TensorId> &args);
  TensorId atan(const std::vector<TensorId> &args);
  TensorId atanh(const std::vector<TensorId> &args);
  TensorId cast(const std::vector<TensorId> &args);
  TensorId ceil(const std::vector<TensorId> &args);
  TensorId cos(const std::vector<TensorId> &args);
  TensorId cosh(const std::vector<TensorId> &args);
  TensorId div(const std::vector<TensorId> &args);
  TensorId elu(const std::vector<TensorId> &args);
  TensorId equal(const std::vector<TensorId> &args);
  TensorId exp(const std::vector<TensorId> &args);
  TensorId floor(const std::vector<TensorId> &args);
  TensorId greater(const std::vector<TensorId> &args);
  TensorId identity(const std::vector<TensorId> &args);
  TensorId less(const std::vector<TensorId> &args);
  TensorId log(const std::vector<TensorId> &args);
  TensorId max(const std::vector<TensorId> &args);
  TensorId mean(const std::vector<TensorId> &args);
  TensorId min(const std::vector<TensorId> &args);
  TensorId mul(const std::vector<TensorId> &args);
  TensorId neg(const std::vector<TensorId> &args);
  TensorId logical_not(const std::vector<TensorId> &args);
  TensorId logical_or(const std::vector<TensorId> &args);
  TensorId pow(const std::vector<TensorId> &args);
  TensorId reciprocal(const std::vector<TensorId> &args);
  TensorId relu(const std::vector<TensorId> &args);
  TensorId sigmoid(const std::vector<TensorId> &args);
  TensorId sin(const std::vector<TensorId> &args);
  TensorId sinh(const std::vector<TensorId> &args);
  TensorId softsign(const std::vector<TensorId> &args);
  TensorId sqrt(const std::vector<TensorId> &args);
  TensorId sub(const std::vector<TensorId> &args);
  TensorId sum(const std::vector<TensorId> &args);
  TensorId tan(const std::vector<TensorId> &args);
  TensorId tanh(const std::vector<TensorId> &args);
  TensorId logical_xor(const std::vector<TensorId> &args);

  TensorId convolution(const std::vector<TensorId> &args,
                       const std::vector<int> strides,
                       const std::vector<int> padding,
                       const std::vector<int> dilation,
                       int groups = 1);

  TensorId gemm(const std::vector<TensorId> &args,
                float alpha,
                float beta,
                int transA,
                int transB);

  TensorId matmul(const std::vector<TensorId> &args);

  std::string getModelProto() const;

private:
  TensorId add_simple_op(const std::vector<TensorId> &args,
                         const char *name,
                         int arg_count);

  TensorId add_variadic_op(const std::vector<TensorId> &args, const char *name);

  TensorId getNextId();

  uint64_t next_id_;

  onnx::ModelProto model_;
};

} // namespace willow
#endif // GUARD_BUILDER_IMPL_H
