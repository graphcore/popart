#include <popart/typefunctor.hpp>

namespace popart {
namespace typefunctor {

template <> int64_t Int64FromVoid::operator()<popart::Half>(void *) {
  throw error("functor Int64FromVoid cannot handle popart::Half");
}

} // namespace typefunctor
} // namespace popart
