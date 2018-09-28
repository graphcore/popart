#ifndef GUARD_NEURALNET_PATTERNS_HPP
#define GUARD_NEURALNET_PATTERNS_HPP

#include <map>
#include <neuralnet/names.hpp>

namespace neuralnet {

class Op;
class Tensor;

class Pattern {
public:
  Pattern()          = default;
  virtual ~Pattern() = default;

  // Does op at the root of the
  // pattern make a match?
  virtual bool matches(const Op *) const = 0;
  // If the pattern is applied
  // (replaced with another patther),
  // which tensors will be removed?
  virtual std::vector<const Tensor *> removes(const Op *) const = 0;
  // apply the pattern,
  // changes the graph of the op
  virtual void apply(Op *) const = 0;
  // if applied to op, would there
  // be no Anchored tensors removed?
  bool removesNoAnchored(const Op *) const;
};

// remove ()->[] where () is Tensor and [] is an Op and ()->[]
// forms part of [.]->()->[]->(.). after this, this section will
// be [.]->(.). [] is the root op of the pattern. 
// [] IS THE IDENTITY OP
class PreUniRepl : public Pattern {
public:
  PreUniRepl()                   = default;
  virtual ~PreUniRepl() override = default;

  // Pad with pad size zero
  // Sum with one input
  virtual bool matches(const Op *) const override final;
  virtual std::vector<const Tensor *> removes(const Op *) const override final;
  virtual void apply(Op *) const override final;
};

// consider,
// (ori) -> [*] -> {(rep1), (rep2), (rep3)}
// where rep1 = ori, rep2 = ori, rep3 = ori
// We call the Op [*] a N-replicator.
// It is similar to identity above, but it
// has N=3 copies of the input instead of N=1.
// if (ori) -> {[op0], [*], [op1]}, and
// (rep1) -> {[op0], [op2], [op2]}, and
// (rep2) -> {[op2], [op3]}
// (rep3) -> {}
// then this should be replaced by
// (ori) -> {[op0], [op0], [o1], [op2], [op2], [op2], [o3]}
// removals : [*], (rep1), (rep2), (rep3)
class PostNRepl : public Pattern {
public:
  PostNRepl()                   = default;
  virtual ~PostNRepl() override = default;

  // AddGrad (where N = 2)
  // Pad with pad size zero (where N = 1) *1
  // Sum with one input (where N = 1) *1
  virtual bool matches(const Op *) const override final;
  virtual std::vector<const Tensor *> removes(const Op *) const override final;
  virtual void apply(Op *) const override final;

  // *1 : this pattern matches and removes []->() 
  // whereas PreUniRepl matches and removes ()->[]. 
  // This pattern can be considered PostUniRepl when N = 1
};

} // namespace neuralnet

#endif
