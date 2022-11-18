# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import math
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import Literal

import numpy as np
import pytest

import popxl
import popart
import popart._internal.ir as pb_ir
from popxl import ops
import popxl.dtypes as dtypes
from popxl.replica_grouping import ReplicaGrouping
from popxl.tensor import Variable


class TestReplicaGrouping:
    def test_replica_grouping_constructor(self):
        """Test that calling the constructor raises an error."""
        with pytest.raises(RuntimeError):
            popxl.ReplicaGrouping()

    @pytest.mark.parametrize(
        "num_replicas, stride, group_size, assignment",
        [
            (8, 1, 8, [0, 0, 0, 0, 0, 0, 0, 0]),
            (8, 1, 2, [0, 0, 1, 1, 2, 2, 3, 3]),
            (8, 2, 4, [0, 1, 0, 1, 0, 1, 0, 1]),
            (8, 1, 4, [0, 0, 0, 0, 1, 1, 1, 1]),
        ],
    )
    def test_replica_grouping_construct(
        self, num_replicas, stride, group_size, assignment
    ):
        """Test constructing some replica grouping objects."""
        ir = popxl.Ir()
        ir.replication_factor = num_replicas
        rg = ir.replica_grouping(stride=stride, group_size=group_size)
        assert rg.stride == stride
        assert rg.group_size == group_size
        assert rg.assignment == assignment
        assert isinstance(rg.num_groups, int)
        assert rg.const_rg
        assert rg.to_device_map is None
        assert rg.to_host_map is None
        assert isinstance(rg._to_variable_settings(), popart.VariableSettings)
        assert isinstance(rg._pb_replica_grouping, pb_ir.ReplicaGrouping)

    def test_replica_grouping_repr(self):
        ir = popxl.Ir()
        ir.replication_factor = 8
        rg = ir.replica_grouping(stride=2, group_size=4)
        assert (
            repr(rg)
            == "ReplicaGrouping(num_replicas=8, stride=2, group_size=4, num_groups=2)"
        )

    def test_eq(self):
        ir = popxl.Ir()
        ir.replication_factor = 8
        assert ir.replica_grouping(stride=2, group_size=4) == ir.replica_grouping(
            stride=2, group_size=4
        )
        assert ir.replica_grouping(stride=2, group_size=2) != ir.replica_grouping(
            stride=2, group_size=4
        )


# Some examples of the expected variable settings.
stride_and_size_examples = [
    # ---------------------
    {
        "replicas": 2,
        # ->
        "inputs": {
            "stride": 1,
            "group_size": 2,
        },
    },
    # ---------------------
    {
        "replicas": 4,
        # ->
        "inputs": {
            "stride": 2,
            "group_size": 2,
        },
    },
    # ----------------------
    {
        "replicas": 4,
        # ->
        "inputs": {
            "stride": 1,
            "group_size": 2,
        },
    },
    # ----------------------
    {
        "replicas": 16,
        # ->
        "inputs": {
            "stride": 2,
            "group_size": 8,
        },
    },
    # ---------------------- Note: VariableSettings for TMP_4 on a pod256
    {
        "replicas": 256,
        # ->
        "inputs": {
            "stride": 64,
            "group_size": 4,
        },
    },
    # -------------------------- Default values
    {
        "replicas": 256,
        # ->
        "inputs": {},
    },
]

CHANNELS = 3
DATA_LEN = 7
O_DIM = 5

input_shape = [O_DIM, CHANNELS, DATA_LEN, DATA_LEN]


@pytest.mark.parametrize("settings", stride_and_size_examples)
# Test every config with both retrieval modes
@pytest.mark.parametrize("retrieval_mode", ["one_per_group", "all_replicas"])
class TestVariableReplicaGrouping:
    """Testing ReplicaGrouping vs. expected VariableSettings"""

    def _verify_settings(self, replica_grouping: ReplicaGrouping):
        """Verify the settings against the provided variable.

        Args:
            replica_grouping: The grouping to test.
        """
        variable_settings = replica_grouping._to_variable_settings()
        assert variable_settings.isUsingCommGroup() is False
        assert variable_settings.getStride() == replica_grouping.stride
        assert variable_settings.getGroupSize() == replica_grouping.group_size

    def _get_weights_array(
        self, shape: List[int], replica_grouping: Optional[ReplicaGrouping] = None
    ) -> np.ndarray:
        """Get a correctly shaped numpy array given the provided arguments.

        Args:
            shape (List[int]): The tensor shape on device.
            replica_grouping (ReplicaGrouping) The replica grouping used to create
                the variable. Optional.

        Returns:
            np.array: An np.array of the correct size for use in this case.
        """
        reshape = []
        if replica_grouping.num_groups > 1:
            reshape = [replica_grouping.num_groups]
        reshape.extend(shape)

        array = np.random.random_sample(reshape).astype(np.float32)
        return array

    def _get_remote_buffer(
        # pylint: disable=unused-argument
        self,
        shape: List[int],
        replicas: int = 1,
        group_size=0,
        stride=1,
    ) -> np.ndarray:  # pylint: disable=unused-argument
        reshape = []
        num_groups = 1 if group_size == 0 else replicas // group_size
        if num_groups > 1:
            reshape = [num_groups]
        reshape.extend(shape)

        return popxl.RemoteBuffer(reshape, dtypes.float, num_groups)

    def _verify_variable(self, settings: Dict[str, Any], var: Variable) -> None:
        """Verify the variable provided has correct shape given the provided settings.

        Args:
            settings (Dict[str, Any]): The settings to verify against
            var (Variable): The variable to check.
        """
        if "group_size" not in settings["inputs"].keys():
            # Default group_size is all the replicas
            group_size = settings["replicas"]
        else:
            group_size = settings["inputs"]["group_size"]
        num_groups = settings["replicas"] // group_size

        if var.retrieval_mode == "one_per_group":
            assert var.shape == (*input_shape,)
        elif num_groups > 1:
            assert var.shape_on_host == (
                num_groups,
                *input_shape,
            )
        else:
            assert var.shape == (*input_shape,)

    def test_variable_settings_object(
        self,
        settings: Dict[str, Any],
        retrieval_mode: Literal["one_per_group", "all_replicas"],
    ):  # pylint: disable=unused-argument
        """Test the PopXL replica groupings agree with the expected popart VariableSettings.

        Args:
            settings (Dict[str, Any]): Some examples of the expected variable settings.
            retrieval_mode (Literal["one_per_group", "all_replicas"]):
                One of:
                - "one_per_group": Return only the first replica's variable per group.
                - "all_replicas": Return all replica's variables in every group.
                Unused in this test, kept here to keep the parameterized tests tidy.
        """
        ir = popxl.Ir()
        ir.replication_factor = settings["replicas"]
        main = ir.main_graph

        with main:
            replica_grouping = ir.replica_grouping(**settings["inputs"])
            self._verify_settings(replica_grouping)

    def test_variable_settings_normal_variable(
        self,
        settings: Dict[str, Any],
        retrieval_mode: Literal["one_per_group", "all_replicas"],
    ):
        """Test the variable is of the expected shape when constructing a variable
        using a replica grouping.

        Args:
            settings (Dict[str, Any]): Some examples of the expected variable settings.
            retrieval_mode (Literal["one_per_group", "all_replicas"]):
                One of:
                - "one_per_group": Return only the first replica's variable per group.
                - "all_replicas": Return all replica's variables in every group.
        """
        ir = popxl.Ir()
        ir.replication_factor = settings["replicas"]
        main = ir.main_graph

        with main:
            replica_grouping = ir.replica_grouping(**settings["inputs"])
            x1 = self._get_weights_array(input_shape, replica_grouping)
            v1 = popxl.variable(
                x1,
                name="v1",
                replica_grouping=replica_grouping,
                retrieval_mode=retrieval_mode,
            )

            print("Shape : ", v1.shape)
            self._verify_variable(settings, v1)

    def test_variable_settings_remote_variable(
        self,
        settings: Dict[str, Any],
        retrieval_mode: Literal["one_per_group", "all_replicas"],
    ):
        """Test the variable is of the expected shape when constructing a remote_variable
        using a replica grouping.

        Args:
            settings (Dict[str, Any]): Some examples of the expected variable settings.
            retrieval_mode (Literal["one_per_group", "all_replicas"]):
                One of:
                - "one_per_group": Return only the first replica's variable per group.
                - "all_replicas": Return all replica's variables in every group.
        """
        ir = popxl.Ir()
        ir.replication_factor = settings["replicas"]
        main = ir.main_graph

        with main:
            replica_grouping = ir.replica_grouping(**settings["inputs"])
            x1 = self._get_weights_array(input_shape, replica_grouping)
            buffer = popxl.RemoteBuffer(
                x1[0, ...].shape, dtypes.float, settings["replicas"]
            )

            v1 = popxl.remote_variable(
                x1,
                buffer,
                0,
                name="v1",
                replica_grouping=replica_grouping,
                retrieval_mode=retrieval_mode,
            )

            self._verify_variable(settings, v1)

    def test_variable_settings_replica_sharded_variable(
        self,
        settings: Dict[str, Any],
        retrieval_mode: Literal["one_per_group", "all_replicas"],
    ):
        """Test the variable is of the expected shape when constructing a replica_sharded_variable
        using a replica grouping.

        Args:
            settings (Dict[str, Any]): Some examples of the expected variable settings.
            retrieval_mode (Literal["one_per_group", "all_replicas"]):
                One of:
                - "one_per_group": Return only the first replica's variable per group.
                - "all_replicas": Return all replica's variables in every group.
        """
        ir = popxl.Ir()
        ir.replication_factor = settings["replicas"]
        main = ir.main_graph

        with main:
            replica_grouping = ir.replica_grouping(**settings["inputs"])
            x1 = self._get_weights_array(input_shape, replica_grouping)
            v1, _ = popxl.replica_sharded_variable(
                x1,
                dtypes.float,
                name="v1",
                replica_grouping=replica_grouping,
                retrieval_mode=retrieval_mode,
            )

            self._verify_variable(
                settings, v1
            )  # note we verify vs. the whole tensor, not the shard.

    def test_variable_settings_remote_replica_sharded_variable(
        self,
        settings: Dict[str, Any],
        retrieval_mode: Literal["one_per_group", "all_replicas"],
    ):
        """Test the variable is of the expected shape when constructing a remote_replica_sharded_variable
        using a replica grouping.

        Args:
            settings (Dict[str, Any]): Some examples of the expected variable settings.
            retrieval_mode (Literal["one_per_group", "all_replicas"]):
                One of:
                - "one_per_group": Return only the first replica's variable per group.
                - "all_replicas": Return all replica's variables in every group.
        """
        ir = popxl.Ir()
        ir.replication_factor = settings["replicas"]
        main = ir.main_graph

        with main:
            replica_grouping = ir.replica_grouping(**settings["inputs"])
            x1 = self._get_weights_array(input_shape, replica_grouping)

            if "group_size" in settings["inputs"].keys():
                var_shard_shape: Tuple[int, ...] = (
                    math.ceil(x1[0, ...].size / settings["inputs"]["group_size"]),
                )
            else:
                var_shard_shape = x1[0, ...].shape
            buffer = popxl.RemoteBuffer(var_shard_shape, dtypes.float, entries=1)

            v1 = popxl.remote_replica_sharded_variable(
                x1,
                buffer,
                0,
                name="v1",
                replica_grouping=replica_grouping,
                retrieval_mode=retrieval_mode,
            )

            self._verify_variable(settings, v1)


transpose_examples = [
    {
        "input": {
            "stride": 1,
            "group_size": 1,
            "assignment": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        },
        "transpose": {
            "stride": 1,
            "group_size": 16,
            "assignment": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        },
    },
    {
        "input": {
            "stride": 1,
            "group_size": 4,
            "assignment": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3],
        },
        "transpose": {
            "stride": 4,
            "group_size": 4,
            "assignment": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
        },
    },
    {
        "input": {
            "stride": 2,
            "group_size": 8,
            "assignment": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        },
        "transpose": {
            "stride": 1,
            "group_size": 2,
            "assignment": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7],
        },
    },
    {
        "input": {
            "stride": 8,
            "group_size": 2,
            "assignment": [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7],
        },
        "transpose": {
            "stride": 1,
            "group_size": 8,
            "assignment": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
        },
    },
]


def test_transpose():
    ir = popxl.Ir(replication=16)
    for i, example in enumerate(transpose_examples):
        input = example["input"]
        rg = ir.replica_grouping(input["stride"], input["group_size"])
        assert rg.assignment == input["assignment"], f"example {i}"

        transpose = example["transpose"]
        rg_transpose = rg.transpose()
        assert rg_transpose.stride == transpose["stride"], f"example {i}"
        assert rg_transpose.group_size == transpose["group_size"], f"example {i}"
        assert rg_transpose.assignment == transpose["assignment"], f"example {i}"


transpose_from_assignment_examples = [
    {
        "input": [0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3],
        "transpose": [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2],
    },
    {
        "input": [0, 0, 1, 1, 0, 0, 1, 1],
        "transpose": [0, 1, 0, 1, 2, 3, 2, 3],
    },
]


def test_transpose_from_assignments():
    for i, example in enumerate(transpose_from_assignment_examples):
        ir = popxl.Ir(len(example["input"]))
        rg = ir.replica_grouping_from_assignments(example["input"])

        rg_transpose = rg.transpose()
        assert rg_transpose.assignment == example["transpose"], f"example {i}"


class TestReplicaGroupingNonConst:
    def test_transpose_from_const_to_nonconst(self):
        ir = popxl.Ir(replication=8)
        rg = ir.replica_grouping(stride=2, group_size=2)
        assert rg.assignment == [0, 1, 0, 1, 2, 3, 2, 3]

        rg_T = rg.transpose()
        assert not rg_T.is_const
        assert rg_T.assignment == [0, 0, 1, 1, 0, 0, 1, 1]

    def test_transpose_from_nonconst_to_const(self):
        ir = popxl.Ir(replication=8)
        rg = ir.replica_grouping_from_assignments([0, 0, 1, 1, 0, 0, 1, 1])

        rg_T = rg.transpose()
        assert rg_T.is_const
        assert rg_T.assignment == [0, 1, 0, 1, 2, 3, 2, 3]

    def test_transpose_from_nonconst_to_nonconst(self):
        ir = popxl.Ir(replication=8)
        rg = ir.replica_grouping_from_assignments([0, 1, 0, 1, 2, 3, 2, 3])

        rg_T = rg.transpose()
        assert not rg_T.is_const
        assert rg_T.assignment == [0, 0, 1, 1, 0, 0, 1, 1]

    def test_constructor_checks(self):
        # Wrong length
        ir = popxl.Ir(replication=4)
        with pytest.raises(ValueError):
            ir.replica_grouping_from_assignments([0])

        # Different group sizes
        with pytest.raises(ValueError):
            ir.replica_grouping_from_assignments([0, 0, 0, 1])

        # Group indexes dont start at 0
        with pytest.raises(ValueError):
            ir.replica_grouping_from_assignments([1, 1, 2, 2])

        # Group indexes are not consecutive
        with pytest.raises(ValueError):
            ir.replica_grouping_from_assignments([0, 0, 2, 2])

        # Group indexes are not monotonic
        with pytest.raises(ValueError):
            ir.replica_grouping_from_assignments([1, 1, 0, 0])

    def test_properties(self):
        ir = popxl.Ir(replication=8)
        rg = ir.replica_grouping_from_assignments([0, 0, 1, 1, 0, 0, 1, 1])

        assert rg.assignment == [0, 0, 1, 1, 0, 0, 1, 1]
        assert rg.group_size == 4
        assert rg.num_groups == 2
        assert not rg.is_const
        assert isinstance(rg.num_groups, int)
        assert rg.to_device_map
        assert rg.to_host_map

        with pytest.raises(NotImplementedError):
            rg.stride

        with pytest.raises(NotImplementedError):
            rg._to_variable_settings()

        with pytest.raises(NotImplementedError):
            rg._pb_replica_grouping

    def test_from_assignment_with_const_stride(self):
        ir = popxl.Ir(replication=8)
        rg = ir.replica_grouping_from_assignments([0, 0, 0, 0, 1, 1, 1, 1])

        assert rg.is_const
        assert rg.group_size == 4
        assert rg.stride == 1


nonconst_variable_examples = [
    {
        "assignment": [0, 0, 1, 1, 0, 0, 1, 1],
        "const_assignment": [0, 0, 1, 1, 2, 2, 3, 3],
    },
    {
        "assignment": [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
        "const_assignment": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    },
    {
        "assignment": [0, 1, 1, 0, 0, 1, 1, 0],
        "const_assignment": [0, 1, 2, 3, 4, 5, 6, 7],
    },
    {
        "assignment": [0, 1, 2, 3, 1, 2, 3, 0],
        "const_assignment": [0, 1, 2, 3, 4, 5, 6, 7],
    },
    {
        "assignment": [0, 1, 0, 0, 1, 1, 0, 1],
        "const_assignment": [0, 1, 2, 3, 4, 5, 6, 7],
    },
]


def test_nonconst_rg_variable():
    for i, example in enumerate(nonconst_variable_examples):
        assignment = example["assignment"]
        const_assignment = example["const_assignment"]
        assert len(assignment) == len(const_assignment)
        replication = len(assignment)
        num_groups = len(set(assignment))

        ir = popxl.Ir(replication=replication)
        data = np.arange(num_groups * 3 * 2).reshape(num_groups, 3, 2).astype("int32")
        rg = ir.replica_grouping_from_assignments(assignment)
        assert not rg.is_const, f"example {i}"
        assert rg.const_rg.assignment == const_assignment, f"example {i}"

        with ir.main_graph:
            var = popxl.variable(data, popxl.int32, name="var", replica_grouping=rg)
            assert var.shape == (3, 2)
            ops.print_tensor(var)  # Stops the variable being optimised away

        with popxl.Session(ir, "ipu_model") as session:
            device_data = session.get_tensor_data(var)
            np.testing.assert_array_equal(device_data, data, err_msg=f"example {i}")

            data_rev = data[::-1]
            session.write_variable_data(var, data_rev)
            device_data = session.get_tensor_data(var)
            np.testing.assert_array_equal(device_data, data_rev, err_msg=f"example {i}")
