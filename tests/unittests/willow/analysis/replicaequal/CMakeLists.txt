# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
add_unit_test(unittest_willow_analysis_replicaequal_replicaequalanalysis test_replicaequalanalysis.cpp SUPPORT_LIBS ir-query-test-util test-graphs-test-util)
add_unit_test(unittest_willow_analysis_replicaequal_replicaequalanalysisresults test_replicaequalanalysisresults.cpp SUPPORT_LIBS test-graphs-test-util)
