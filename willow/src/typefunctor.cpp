#include <poponnx/typefunctor.hpp>

namespace poponnx {
namespace typefunctor {

template <> int64_t Int64FromVoid::operator()<poponnx::Half>(void *) {
  throw error("functor Int64FromVoid cannot handle poponnx::Half");
}

} // namespace typefunctor
} // namespace poponnx
