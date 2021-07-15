// Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#define BOOST_TEST_MODULE Speed0SubgraphTest

#include "validate.hpp"
#include <../random_util.hpp>
#include <boost/test/unit_test.hpp>
#include <chrono>
#include <iostream>
#include <vector>

#include "blip.hpp"

using namespace fwtools::subgraph;
using namespace blip;

std::vector<std::unique_ptr<Blip>> getBlips(int taskId, int n_nodes) {
  std::vector<std::unique_ptr<Blip>> blips;
  if (taskId == 0) {
    for (int i = 0; i < n_nodes; ++i) {
      // 0 0 0 0 0 0 ....
      blips.emplace_back(std::unique_ptr<Blip>(new Blip(0, 10.0f, {})));
    }
  }

  else if (taskId == 1) {
    // 0 1 2 3 4 5 6  ... N 0 1 2 3 4 5 6 ... N
    for (int i = 0; i < n_nodes; ++i) {
      blips.emplace_back(
          std::unique_ptr<Blip>(new Blip(i % (n_nodes / 2), 10.0f, {})));
    }
  }

  else if (taskId == 2) {
    // random
    int seed = 1011;
    popart::DefaultRandomEngine eng(seed);
    popart::UniformIntDistribution<int> idis(0, 2);
    for (int i = 0; i < n_nodes; ++i) {
      auto symbol = idis(eng);
      blips.emplace_back(
          std::unique_ptr<Blip>(new Blip(symbol, 10.0f + symbol, {})));
    }
  }

  else if (taskId == 3) {
    // 0 ... 37 0 ... 37 0 ... 37
    for (int i = 0; i < n_nodes; ++i) {
      blips.emplace_back(std::unique_ptr<Blip>(new Blip(i % (37), 10.0f, {})));
    }
  }

  else {
    throw std::runtime_error("Invalid taskId");
  }

  return blips;
}

class AlgoStats {
public:
  std::vector<Match> matches;
  float timeMs;
  int nMatches;
};

AlgoStats getStats(int taskId,
                   int n_nodes,
                   const std::vector<const Blip *> &sched,
                   OutlinerAlgorithm algo) {

  std::vector<std::pair<size_t, size_t>> sequences(sched.size());
  float sequenceBreakCost = 0.0f;

  using namespace std::chrono;

  float threshold = -1.0;
  auto t0         = steady_clock::now();
  auto matches0   = getRinseMatches<const Blip>(
      sched, sequences, threshold, sequenceBreakCost, algo);
  auto t1 = steady_clock::now();
  auto d0 = std::chrono::duration<double, std::milli>(t1 - t0).count();

  AlgoStats stats;
  stats.matches  = matches0;
  stats.timeMs   = d0;
  stats.nMatches = matches0.size();

  return stats;
}

BOOST_AUTO_TEST_CASE(Speed0_Subgraph) {

  auto test = [](OutlinerAlgorithm algo, int n_nodes, int taskId) {
    auto blips = getBlips(taskId, n_nodes);
    std::vector<const Blip *> sched;
    for (int i = 0; i < n_nodes; ++i) {
      sched.push_back(blips[i].get());
    }
    auto stats1 = getStats(taskId, n_nodes, sched, algo);
    std::cout << n_nodes << " Blips : " << stats1.timeMs << " [ms], "
              << stats1.nMatches << " matches." << std::endl;

    BOOST_CHECK(isValid(stats1.matches, sched) == true);
  };

  std::cout << "\nTask 0 : 000000000....000000000" << std::endl;
  std::cout << "-------------------------------" << std::endl;
  std::cout << "Algorithm 0" << std::endl;
  test(OutlinerAlgorithm::ALGO0, 32, 0);
  test(OutlinerAlgorithm::ALGO0, 64, 0);
  test(OutlinerAlgorithm::ALGO0, 128, 0);

  std::cout << "\nAlgorithm 1" << std::endl;
  test(OutlinerAlgorithm::ALGO1, 32, 0);
  test(OutlinerAlgorithm::ALGO1, 64, 0);
  test(OutlinerAlgorithm::ALGO1, 128, 0);
  test(OutlinerAlgorithm::ALGO1, 256, 0);

  std::cout << "\nTask 1 : 0 1 2 ... N 0 1 2 .. N" << std::endl;
  std::cout << "-------------------------------" << std::endl;
  std::cout << "Algorithm 0" << std::endl;
  test(OutlinerAlgorithm::ALGO0, 32, 1);
  test(OutlinerAlgorithm::ALGO0, 64, 1);
  test(OutlinerAlgorithm::ALGO0, 128, 1);

  // Linear : time for Ukkonen to construct the suffix tree
  std::cout << "\nAlgorithm 1" << std::endl;
  test(OutlinerAlgorithm::ALGO1, 512, 1);
  test(OutlinerAlgorithm::ALGO1, 1024, 1);
  test(OutlinerAlgorithm::ALGO1, 2048, 1);
  test(OutlinerAlgorithm::ALGO1, 4096, 1);

  std::cout << "\nTask 2 : uniform random 0/1/2" << std::endl;
  std::cout << "------------------------------" << std::endl;
  std::cout << "Algorithm 0" << std::endl;
  test(OutlinerAlgorithm::ALGO0, 256, 2);
  test(OutlinerAlgorithm::ALGO0, 512, 2);
  test(OutlinerAlgorithm::ALGO0, 1024, 2);

  std::cout << "\nAlgorithm 1" << std::endl;
  test(OutlinerAlgorithm::ALGO1, 256, 2);
  test(OutlinerAlgorithm::ALGO1, 512, 2);
  test(OutlinerAlgorithm::ALGO1, 1024, 2);
  test(OutlinerAlgorithm::ALGO1, 2048, 2);

  std::cout << "\nTask 3 : 0 ... 37 0 ... 37 0 ... 37 etc" << std::endl;
  std::cout << "---------------------------------------" << std::endl;
  std::cout << "Algorithm 0" << std::endl;
  test(OutlinerAlgorithm::ALGO0, 32, 3);
  test(OutlinerAlgorithm::ALGO0, 64, 3);
  test(OutlinerAlgorithm::ALGO0, 128, 3);

  std::cout << "\nAlgorithm 1" << std::endl;
  test(OutlinerAlgorithm::ALGO1, 256, 3);
  test(OutlinerAlgorithm::ALGO1, 512, 3);
  test(OutlinerAlgorithm::ALGO1, 1024, 3);
  test(OutlinerAlgorithm::ALGO1, 2048, 3);
}
