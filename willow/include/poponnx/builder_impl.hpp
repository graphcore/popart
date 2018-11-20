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

  std::string addInputTensor(const TensorInfo &tensorInfo);

  void addOutputTensor(const std::string &arg0);

  // Operations requiring only tensor inputs
  std::string abs(const std::vector<std::string> &args);
  std::string acos(const std::vector<std::string> &args);
  std::string acosh(const std::vector<std::string> &args);
  std::string add(const std::vector<std::string> &args);
  std::string logical_and(const std::vector<std::string> &args);
  std::string asin(const std::vector<std::string> &args);
  std::string asinh(const std::vector<std::string> &args);
  std::string atan(const std::vector<std::string> &args);
  std::string atanh(const std::vector<std::string> &args);
  std::string cast(const std::vector<std::string> &args);
  std::string ceil(const std::vector<std::string> &args);
  std::string cos(const std::vector<std::string> &args);
  std::string cosh(const std::vector<std::string> &args);
  std::string div(const std::vector<std::string> &args);
  std::string elu(const std::vector<std::string> &args);
  std::string equal(const std::vector<std::string> &args);
  std::string exp(const std::vector<std::string> &args);
  std::string floor(const std::vector<std::string> &args);
  std::string greater(const std::vector<std::string> &args);
  std::string identity(const std::vector<std::string> &args);
  std::string less(const std::vector<std::string> &args);
  std::string log(const std::vector<std::string> &args);
  std::string max(const std::vector<std::string> &args);
  std::string mean(const std::vector<std::string> &args);
  std::string min(const std::vector<std::string> &args);
  std::string mul(const std::vector<std::string> &args);
  std::string neg(const std::vector<std::string> &args);
  std::string logical_not(const std::vector<std::string> &args);
  std::string logical_or(const std::vector<std::string> &args);
  std::string pow(const std::vector<std::string> &args);
  std::string reciprocal(const std::vector<std::string> &args);
  std::string relu(const std::vector<std::string> &args);
  std::string sigmoid(const std::vector<std::string> &args);
  std::string sin(const std::vector<std::string> &args);
  std::string sinh(const std::vector<std::string> &args);
  std::string softsign(const std::vector<std::string> &args);
  std::string sqrt(const std::vector<std::string> &args);
  std::string sub(const std::vector<std::string> &args);
  std::string sum(const std::vector<std::string> &args);
  std::string tan(const std::vector<std::string> &args);
  std::string tanh(const std::vector<std::string> &args);
  std::string logical_xor(const std::vector<std::string> &args);

  std::string convolution(const std::vector<std::string> &args,
                          const std::vector<int> strides,
                          const std::vector<int> padding,
                          const std::vector<int> dilation,
                          int groups = 1);

  std::string gemm(const std::vector<std::string> &args,
                   float alpha,
                   float beta,
                   int transA,
                   int transB);

  std::string matmul(const std::vector<std::string> &args);

  std::string getModelProto() const;

private:
  std::string add_simple_op(const std::vector<std::string> &args,
                            const char *name,
                            int arg_count);

  std::string add_variadic_op(const std::vector<std::string> &args,
                              const char *name);

  std::string getNextId();

  uint64_t next_id_;

  onnx::ModelProto model_;
};

} // namespace willow
#endif // GUARD_BUILDER_IMPL_H
