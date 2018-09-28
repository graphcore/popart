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
class Identity : public Pattern {
public:
  Identity()                   = default;
  virtual ~Identity() override = default;

  // Pad with pad size zero
  // Sum with one input
  virtual bool matches(const Op *) const override final;
  virtual std::vector<const Tensor *> removes(const Op *) const override final;
  virtual void apply(Op *) const override final;
};

} // namespace neuralnet

#endif
