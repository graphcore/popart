// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_LAMBSERIALISEDWEIGHTPATTERN_HPP
#define GUARD_NEURALNET_LAMBSERIALISEDWEIGHTPATTERN_HPP

#include <string>
#include <vector>
#include <popart/patterns/pattern.hpp>

#include "popart/names.hpp"

namespace popart {

class SumOp;
class Graph;
class Op;
class Tensor;

/**
 * This Pattern finds Weights that have been serialised and are being
 * updated in the Lamb Optimizer in slices. Transforming:
 *
 *   Slice(W)        U_sliced    }
 *     | (R1)           | (R2)   }
 *   ReduceScatter      |        }   (Optional, to support RTS)
 *     |                |        }
 *   LambSquare     LambSquare   }   x N
 *       |              |        }
 *     AllReduce   AllReduce     }   (Optional, to support RTS)
 *           \      /            }
 *         AdamVarUpdate         }
 * Into:
 *
 *   Slice(W)        U_sliced    }
 *     |                |        }
 *   ReduceScatter      |        }   (Optional, to support RTS)
 *     |                |        }
 *   LambSquare     LambSquare   }   x N
 *     |                |        }
 *     Sum             Sum
 *       \            /
 *     AllReduce   AllReduce     }   (Optional, to support RTS)
 *           \      /            }
 *       AdamVarUpdate          }   x N
 *
 * A key property of LambSquare is that the output has not been sqrt yet, so it
 * is valid to just Sum the outputs.
 */
class LambSerialisedWeightPattern : public PreAliasPattern {
public:
  bool matches(Op *) const final;
  std::vector<const Tensor *> touches(Op *) const final;
  bool apply(Op *) const final;

private:
  SumOp *insertSumOp(Graph &graph,
                     std::vector<TensorId> &in_ids,
                     TensorId out_id,
                     Op *ref_op,
                     std::string debug_name) const;
};

} // namespace popart

#endif
