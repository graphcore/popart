// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#include <popart/typefunctor.hpp>

#include "popart/error.hpp"
#include "popart/logging.hpp"

namespace popart {
class Half;

namespace typefunctor {

template <> int64_t Int64FromVoid::operator()<popart::Half>(void *) {
  throw error("functor Int64FromVoid cannot handle popart::Half");
}

} // namespace typefunctor
} // namespace popart
