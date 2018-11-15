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

  std::string add(const std::string &arg0, const std::string &arg1);

  std::string convolution(const std::string &arg0,
                          const std::string &arg1,
                          const std::vector<int> strides,
                          const std::vector<int> padding,
                          const std::vector<int> dilation,
                          int groups = 1);

  std::string convolutionWithBias(const std::string &arg0,
                                  const std::string &arg1,
                                  const std::string &arg2,
                                  const std::vector<int> strides,
                                  const std::vector<int> padding,
                                  const std::vector<int> dilation,
                                  int groups = 1);

  std::string gemm(const std::string &arg0,
                   const std::string &arg1,
                   const std::string &arg2,
                   float alpha,
                   float beta,
                   int transA,
                   int transB);

  std::string matmul(const std::string &arg0, const std::string &arg1);

  std::string getModelProto() const;

private:
  std::string getNextId();

  uint64_t next_id_;

  onnx::ModelProto model_;
};

} // namespace willow
#endif // GUARD_BUILDER_IMPL_H
