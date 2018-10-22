#ifndef GUARD_NEURALNET_ONNXUTIL_HPP
#define GUARD_NEURALNET_ONNXUTIL_HPP

#pragma clang diagnostic push // start ignoring warnings
#pragma clang diagnostic ignored "-Weverything"
#include <onnx/onnx_pb.h>
#pragma clang diagnostic pop // stop ignoring warnings

namespace willow {
namespace onnxutil {

// for many types (float16, float, int, etc) onnx::TensorProto has
// 2 ways of storing the data: either in field raw_data or a field
// specific to the type. This function handles these 2 possibilities.
const void *getData(const onnx::TensorProto &tp);

} // namespace onnxutil
} // namespace willow

#endif
