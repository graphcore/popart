// Copyright (c) 2022 Graphcore Ltd. All rights reserved.
#ifndef POPART_WILLOW_SRC_ENGINEOPTIONSCREATOR_HPP_
#define POPART_WILLOW_SRC_ENGINEOPTIONSCREATOR_HPP_

#include <popart/sessionoptions.hpp>

#include <poplar/EngineOptions.hpp>
#include <poplar/OptionFlags.hpp>
#include <poplar/Target.hpp>

namespace popart {

/**
 * Class for constructing a Poplar EngineOptions object. PopART derives an
 * poplar::OptionsFlags object from the user's SessionOptions and combines this
 * with a poplar::Target to create a poplar::EngineOptions object.
 *
 * The EngineOptions class has some internal logic that is not transparent to
 * PopART. For example, it reads the POPLAR_ENGINE_OPTIONS environment variable
 * and may change the option selection based on this. The hash of the
 * EngineOptions is what we should include in our engine cache hash calculation,
 * as it takes into account all engine options that affect compilation as
 * actually set by the user -- by whichever mechanism.
 *
 * TODO T62481: Once T62481 is done we could use getEngineOptions and pass
 * the EngineOptions object when creating an Engine.
 **/
class EngineOptionsCreator {
public:
  /**
   * Construct an EngineOptionConstructor object.
   * \param sessionOptions The user's SessionOptions.
   * \param target The target.
   * \param replicationFactor The replication factor.
   */
  EngineOptionsCreator(const SessionOptions &sessionOptions,
                       const poplar::Target &target);

  /**
   * Get the option flags that were used to construct the EngineOptions object.
   * \return const OptionFlags& Reference to an OptionFlags object.
   */
  const poplar::OptionFlags &getOptionFlags() const;

  /**
   * Get the EngineOptions object.
   * \return const OptionFlags& Reference to an OptionFlags object.
   */
  const poplar::EngineOptions &getEngineOptions() const;

private:
  // Helper function to derive option flags from session options.
  static poplar::OptionFlags
  deriveOptionFlags(const SessionOptions &sessionOptions);

  // OptionFlags input as determined by PopART as an input to EngineOptions.
  poplar::OptionFlags optionFlags;
  // EngineOptions instance.
  poplar::EngineOptions engineOptions;
};

} // namespace popart

#endif // POPART_WILLOW_SRC_ENGINEOPTIONSCREATOR_HPP_
