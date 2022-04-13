# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import popart
import pytest

import popxl
import popxl.dtypes as dtypes
from popxl.replica_grouping import ReplicaGrouping
from popxl.tensor import Variable


class TestReplicaGrouping:
    def test_replica_grouping_constructor(self):
        """ Test that calling the constructor raises an error. """
        with pytest.raises(RuntimeError):
            popxl.ReplicaGrouping()

    @pytest.mark.parametrize(
        'num_replicas, stride, group_size',
        [
            (8, 1, 8),  # 00000000
            (8, 1, 2),  # 00112233
            (8, 2, 4),  # 01010101
            (8, 1, 4)  # 00001111
        ])
    def test_replica_grouping_construct(self, num_replicas, stride,
                                        group_size):
        """ Test constructing some replica grouping objects. """
        ir = popxl.Ir()
        ir.replication_factor = num_replicas
        rg = ir.replica_grouping(stride=stride, group_size=group_size)
        assert rg.stride == stride
        assert rg.group_size == group_size

    def test_replica_grouping_repr(self):
        ir = popxl.Ir()
        ir.replication_factor = 8
        rg = ir.replica_grouping(stride=2, group_size=4)
        assert repr(
            rg) == 'ReplicaGrouping(num_replicas=8, stride=2, group_size=4)'


# Some examples of the expected variable settings.
stride_and_size_examples = [
    # ---------------------
    {
        "replicas": 2,
        "commType": popart.CommGroupType.All,
        "commSize": 0,
        # ->
        "inputs": {
            "stride": 1,
            "group_size": 2
        }
    },
    # ---------------------
    {
        "replicas": 4,
        "commType": popart.CommGroupType.Orthogonal,
        "commSize": 2,
        # ->
        "inputs": {
            "stride": 2,
            "group_size": 2
        }
    },
    # ----------------------
    {
        "replicas": 4,
        "commType": popart.CommGroupType.Consecutive,
        "commSize": 2,
        # ->
        "inputs": {
            "stride": 1,
            "group_size": 2
        }
    },
    # ----------------------
    {
        "replicas": 16,
        "commType": popart.CommGroupType.Orthogonal,
        "commSize": 2,
        # ->
        "inputs": {
            "stride": 8,
            "group_size": 2
        }
    },
    # ---------------------- Note: VariableSettings for TMP_4 on a pod256
    {
        "replicas": 256,
        "commType": popart.CommGroupType.Orthogonal,
        "commSize": 64,
        # ->
        "inputs": {
            "stride": 4,
            "group_size": 64
        }
    },
    # -------------------------- Default values
    {
        "replicas": 256,
        "commType": popart.CommGroupType.All,
        "commSize": 0,
        # ->
        "inputs": {}
    }
]

CHANNELS = 3
DATA_LEN = 7
O_DIM = 5

input_shape = [O_DIM, CHANNELS, DATA_LEN, DATA_LEN]


@pytest.mark.parametrize("settings", stride_and_size_examples)
class TestVariableReplicaGrouping:
    """Testing ReplicaGrouping vs. expected VariableSettings
    """

    def _verify_settings(self, settings: Dict[str, Any],
                         replica_grouping: ReplicaGrouping):
        """Verify the settings against the provided variable.

        Args:
            settings (Dict[str, Any]): The expected settings.
            v1 (popxl.Variable): The variable to test.
        """
        assert replica_grouping._to_variable_settings(
        ).getSharedVariableDomain().replicaGroupSize == settings["commSize"]
        assert replica_grouping._to_variable_settings(
        ).getSharedVariableDomain().type == settings["commType"]

    def _get_weights_array(self,
                           shape: List[int],
                           replicas: int = 1,
                           group_size=0,
                           stride=1) -> np.ndarray:  # pylint: disable=unused-argument
        """Get a correctly shaped numpy array given the provided arguments.

        Args:
            shape (List[int]): The tensor shape on device.
            replicas (int, optional): Number of replicas used. Defaults to 1.
            group_size (int, optional): Size of the replica groups. Defaults to 0.
            stride (int, optional): Stride of the groups. Unused, argument kept to allow passing
                inputs as kwargs. Defaults to 1.

        Returns:
            np.array: _description_
        """
        reshape = []
        num_groups = replicas // group_size if group_size != 0 else 1
        if num_groups > 1:
            reshape = [num_groups]
        reshape.extend(shape)

        array = np.random.random_sample(reshape).astype(np.float32)
        return array

    def _get_remote_buffer(self,
                           shape: List[int],
                           replicas: int = 1,
                           group_size=0,
                           stride=1) -> np.ndarray:  # pylint: disable=unused-argument
        reshape = []
        num_groups = replicas // group_size if group_size != 0 else 1
        if num_groups > 1:
            reshape = [num_groups]
        reshape.extend(shape)

        return popxl.RemoteBuffer(reshape, dtypes.float, num_groups)

    def _verify_variable(
            self,
            settings: Dict[str, Any],  # pylint: disable=unused-argument
            var: Variable) -> None:
        """Verify the variable provided has correct shape given the provided settings.

        Args:
            settings (Dict[str, Any]): The settings to verify against
            var (Variable): The variable to check.
        """
        assert var.shape == (*input_shape, )

    def test_variable_settings_object(self, settings: Dict[str, Any]):
        """Test the PopXL replica groupings agree with the expected popart VariableSettings.

        Args:
            settings (Dict[str, Any]): Some examples of the expected variable settings.
        """
        ir = popxl.Ir()
        ir.replication_factor = settings["replicas"]
        main = ir.main_graph

        with main:
            replica_grouping = ir.replica_grouping(**settings["inputs"])
            self._verify_settings(settings, replica_grouping)

    def test_variable_settings_normal_variable(self, settings: Dict[str, Any]):
        """Test the variable is of the expected shape when constructing a variable
        using a replica grouping.

        Args:
            settings (Dict[str, Any]): Some examples of the expected variable settings.
        """
        ir = popxl.Ir()
        ir.replication_factor = settings["replicas"]
        main = ir.main_graph

        with main:
            replica_grouping = ir.replica_grouping(**settings["inputs"])
            x1 = self._get_weights_array(input_shape, settings["replicas"],
                                         **settings["inputs"])
            v1 = popxl.variable(x1,
                                name="v1",
                                replica_grouping=replica_grouping)

            print("Shape : ", v1.shape)
            print("Meta shape: ", v1.meta_shape)
            self._verify_variable(settings, v1)

    def test_variable_settings_remote_variable(self, settings: Dict[str, Any]):
        """Test the variable is of the expected shape when constructing a remote_variable
        using a replica grouping.

        Args:
            settings (Dict[str, Any]): Some examples of the expected variable settings.
        """
        ir = popxl.Ir()
        ir.replication_factor = settings["replicas"]
        main = ir.main_graph

        with main:
            replica_grouping = ir.replica_grouping(**settings["inputs"])
            x1 = self._get_weights_array(input_shape, settings["replicas"],
                                         **settings["inputs"])
            buffer = popxl.RemoteBuffer(x1[0, ...].shape, dtypes.float,
                                        settings["replicas"])

            v1 = popxl.remote_variable(x1,
                                       buffer,
                                       0,
                                       name="v1",
                                       replica_grouping=replica_grouping)

            self._verify_variable(settings, v1)

    def test_variable_settings_replica_sharded_variable(
            self, settings: Dict[str, Any]):
        """Test the variable is of the expected shape when constructing a replica_sharded_variable
        using a replica grouping.

        Args:
            settings (Dict[str, Any]): Some examples of the expected variable settings.
        """
        ir = popxl.Ir()
        ir.replication_factor = settings["replicas"]
        main = ir.main_graph

        with main:
            replica_grouping = ir.replica_grouping(**settings["inputs"])
            x1 = self._get_weights_array(input_shape, settings["replicas"],
                                         **settings["inputs"])
            v1, _ = popxl.replica_sharded_variable(
                x1, dtypes.float, name="v1", replica_grouping=replica_grouping)

            self._verify_variable(
                settings,
                v1)  # note we verify vs. the whole tensor, not the shard.

    # @pytest.mark.skip
    def test_variable_settings_remote_replica_sharded_variable(
            self, settings: Dict[str, Any]):
        """Test the variable is of the expected shape when constructing a remote_replica_sharded_variable
        using a replica grouping.

        Args:
            settings (Dict[str, Any]): Some examples of the expected variable settings.
        """
        ir = popxl.Ir()
        ir.replication_factor = settings["replicas"]
        main = ir.main_graph

        with main:
            replica_grouping = ir.replica_grouping(**settings["inputs"])
            x1 = self._get_weights_array(input_shape, settings["replicas"],
                                         **settings["inputs"])

            if "group_size" in settings["inputs"].keys():
                var_shard_shape: Tuple[int, ...] = (math.ceil(
                    x1[0, ...].size / settings["inputs"]["group_size"]), )
            else:
                var_shard_shape = x1[0, ...].shape
            buffer = popxl.RemoteBuffer(var_shard_shape,
                                        dtypes.float,
                                        entries=1)

            v1 = popxl.remote_replica_sharded_variable(
                x1, buffer, 0, name="v1", replica_grouping=replica_grouping)

            self._verify_variable(settings, v1)