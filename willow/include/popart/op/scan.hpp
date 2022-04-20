// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#ifndef GUARD_NEURALNET_SCAN_HPP
#define GUARD_NEURALNET_SCAN_HPP

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>
#include <popart/op.hpp>
#include <popart/op/subgraph.hpp>
#include <popart/tensorindex.hpp>

#include "popart/names.hpp"

namespace popart {
class Graph;
class OpSerialiserBase;
struct OperatorIdentifier;

// Scan operation construct
//
// N state variables (inputs with updated output)
//
//
// Scan op and body inputs:           Loop conversion:
// 0          state variable -----.   -> explicit loop input
// ..         ..                  |      ..
// N-1        state variable --.  |   -> explicit loop input
// N          scan input       |  |   -> implicit loop input
// ..         ..               |  |      ..
// N+M-1      scan input       |  |   -> implicit loop input
// N+M        implicit input   |  |   -> implicit loop input
// ..         ..               |  |      ..
// N+M+L-1    implicit input   |  |   -> implicit loop input
//                             |  |
//                             |  |
// Scan op and body outputs:   |  |
// 0          state variable <-|--'   -> loop/body output
// ..         ..               |         ..
// N-1        state variable <-'      -> loop/body output
// N          scan output             -> loop/body output (+Init explicit input)
// ..         ..                         ..
// N+K-1      scan output             -> loop/body output (+Init explicit input)

class ScanOp : public SubgraphOp {
public:
  ScanOp(const OperatorIdentifier &_opid,
         const Op::Settings &settings_,
         Graph &callee_,
         int numScanInputs_,
         int numImplicitInputs_,
         std::vector<int64_t> scanInputAxes_,
         std::vector<int64_t> scanInputDirections_,
         std::vector<int64_t> scanOutputAxes_,
         std::vector<int64_t> scanOutputDirections_);

  void setup() final;

  void appendOutlineAttributes(OpSerialiserBase &) const override;
  float getSubgraphValue() const final { return getLowSubgraphValue(); }

  std::unique_ptr<Op> clone() const override;

  Graph &getCalledGraph() const override;
  void setCalledGraph(Graph &) override;

  InIndex subgraphInToOpInIndex(InIndex index) const override;
  InIndex opInToSubgraphInIndex(InIndex index) const override;

  OutIndex subgraphOutToOpOutIndex(OutIndex index) const override;
  OutIndex opOutToSubgraphOutIndex(OutIndex index) const override;

  // Returns the number of iterations required to process the inputs
  int getTripCountValue() const;

  // Returns the number of scan inputs (M)
  int getNumScanInputs() const { return numScanInputs; }

  // Returns the number of  variables (N)
  int getNumVariables() const {
    return input->n() - getNumScanInputs() - getNumImplicitInputs();
  }

  // Returns the number of implicit inputs (L)
  int getNumImplicitInputs() const { return numImplicitInputs; }

  // Returns the number of scan outputs (K)
  int getNumScanOutputs() const { return output->n() - getNumVariables(); }

  // Returns the scan input axis of scan input i
  int64_t getScanInputAxis(int i) const;

  // Returns the scan input axis of scan input i
  bool isScanInputReversed(int i) const { return getScanInputDirection(i); }

  // Returns the scan input axis of scan output i
  int64_t getScanOutputAxis(int i) const;

  // Returns the scan input axis of scan output i
  bool isScanOutputReversed(int i) const { return getScanOutputDirection(i); }

  int64_t getScanInputDirection(int i) const;

  int64_t getScanOutputDirection(int i) const;

private:
  std::reference_wrapper<Graph> callee;
  const int numImplicitInputs;
  const int numScanInputs;
  const std::vector<int64_t> baseScanInputAxes;
  const std::vector<int64_t> baseScanInputDirections;
  const std::vector<int64_t> baseScanOutputAxes;
  const std::vector<int64_t> baseScanOutputDirections;
};

} // namespace popart

#endif
