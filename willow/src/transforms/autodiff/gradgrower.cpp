// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#include <transforms/autodiff/gradgrower.hpp>

namespace popart {

GradGrower::GradGrower(AutodiffIrInterface &dep_) : dep(dep_) {}

} // namespace popart