// Copyright (c) 2018 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_INCLUDE_POPART_SCHEDULER_REQUIREOPTIMAL_HPP_
#define POPART_WILLOW_INCLUDE_POPART_SCHEDULER_REQUIREOPTIMAL_HPP_

namespace popart {

// Passed in to Scheduler functions to specify whether the optimal schedule is
// required, or any valid topological ordering is fine.
enum class RequireOptimalSchedule { Yes = true, No = false };

} // namespace popart

#endif // POPART_WILLOW_INCLUDE_POPART_SCHEDULER_REQUIREOPTIMAL_HPP_
