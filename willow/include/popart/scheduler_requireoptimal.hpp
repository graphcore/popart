// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SCHEDULER_REQUIRE_OPTIMAL_HPP
#define GUARD_NEURALNET_SCHEDULER_REQUIRE_OPTIMAL_HPP

namespace popart {

// Passed in to Scheduler functions to specify whether the optimal schedule is
// required, or any valid topological ordering is fine.
enum class RequireOptimalSchedule { Yes = true, No = false };

} // namespace popart

#endif
