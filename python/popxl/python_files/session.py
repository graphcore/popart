# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import os
import math
from contextlib import ExitStack, contextmanager
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import popart
import popart._internal.ir as _ir
from popxl.ir import Ir
from popxl.streams import DeviceToHostStream, HostToDeviceStream
from popxl.tensor import Constant, HostScalarTensor, Variable
from typing_extensions import Literal

from popxl.utils import _acquire_hw_device_with_timeout, _to_device_info, _offline_device_from_str, _popxl_to_numpy, to_numpy
from popxl.tensor import Tensor

d2hStreamBufferMaps = Mapping[DeviceToHostStream, np.ndarray]
h2dStreamBufferMaps = Mapping[HostToDeviceStream, HostScalarTensor]
StreamBufferMaps = Union[h2dStreamBufferMaps, d2hStreamBufferMaps]


class Session:
    def __init__(
            self,
            ir: Ir,
            device_desc: Literal["ipu_hw", "ipu_model", "cpu"] = "cpu",
    ) -> None:
        """
        Construct a session object.

        A runtime session that can execute a PopXL `Ir`.

        .. warning:: The session object takes ownership of the provided Ir and it cannot be modified
        afterwards.

        Initialise a new session.

        Args:
            ir (Ir): The Ir to use for this session.
            device_desc (Literal["ipu_hw", "ipu_model", "cpu"], optional): The type of
                ipu device to use. One of:
                "ipu_hw": Real IPU hardware. Uses `DeviceConnectionType` == `OnDemand` and
                    `DeviceSelectionCriterion` == `Random`.
                "ipu_model": IPU model.
                "cpu": CPU model. Does not support replication.
                Defaults to "cpu".

        Raises:
            RuntimeError: If the desired device could not be acquired.
        """
        self.ir_: Ir = ir

        d2hs = ir.get_all_d2h_streams()

        # NOTE: Harmlessly re-sets batchesPerStep (ir.num_host_transfers setter
        # would already have done it earlier).
        dataFlow = popart.DataFlow(batchesPerStep=ir.num_host_transfers,
                                   anchorTensors={
                                       d2h.tensor_id:
                                       popart.AnchorReturnType("All")
                                       for d2h in d2hs
                                   })
        self._ir.setDataFlow(dataFlow)

        # No caching code here, done in createFromIr
        # No ir->setDeviceInfo, done in createFromIr
        # No ir->setIsPrepared, done in createFromIr
        # User sets device iterations and rf on Ir beforehand
        # SessionOptions set in Ir ctor and manually by user, beforehand

        self._ir.removeIsolatedTensors(True)
        self._ir.removeIsolatedGraphs()
        self._ir.updateVertices()
        self._ir.finalizeOpDebugInfo()

        self._ir.updateVertices()

        self._ir.setPatterns(
            _ir.patterns.Patterns(_ir.patterns.PatternsLevel.Minimal))
        for g in self._ir.getAllGraphs():
            self._ir.applyPreAliasPatterns(g)
        self._ir.updateVertices()

        # Logs to DEBUG
        self._ir.logIr()

        if isinstance(device_desc, popart.DeviceInfo):
            self._device = device_desc
        elif "POPXL_OFFLINE_DEVICE" in os.environ:
            self._device = _offline_device_from_str(
                device_type=os.environ["POPXL_OFFLINE_DEVICE"],
                num_ipus=self._get_ipu_count())
        else:
            self._device = _to_device_info(device_type=device_desc,
                                           num_ipus=self._get_ipu_count(),
                                           use_popdist=self.ir_._use_popdist)

        # Initialise stack of "was attached" states when entering the Session
        # context
        self._was_attached_stack: List[Union[bool, popart.DeviceInfo]] = []

        # Note: This uses the underlying popart .InferenceSession class, but supports BOTH inference
        # and training (via the autodiff transform). We use the  popart.InferenceSession class to
        # avoid any automatic autodiff-like transformations we want to do manually in popxl.
        self._pb_session = popart.InferenceSession.fromIr(
            ir=self._ir, deviceInfo=self._device)

        self._pb_session.prepareDevice(loadEngine=False)

        # If an attached DeviceInfo is passed to the constructor then __enter__ won't setup the device
        if self.is_attached:
            self.weights_from_host()

    # Methods:

    def _assert_attached_before_runtime(self):
        if not self.is_attached:
            raise ValueError(
                'Must be attached to device before calling a runtime function. Put the call inside a `Session` context, for example `with session: session.run()`.'
            )

    def run_with_outputs(
            self,
            inputs: Optional[h2dStreamBufferMaps] = None,
            outputs: Optional[d2hStreamBufferMaps] = None,
            downcast_inputs: bool = True,
    ) -> None:
        """Run this session with the provided inputs and outputs.

        Inputs will be used as inputs to the model, and outputs will be written to by the session.

        Args:
            inputs (h2dStreamBufferMaps, optional): The inputs to the model. Defaults to None.
            outputs (d2hStreamBufferMaps, optional): The output buffers, these will be written to
                and modified. Defaults to None.
            downcast_inputs (bool): If True 64-bit float/ints inputs will be downcast to 32-bit variants. Defaults to True.

        Raises:
            ValueError: If not attached to device before calling this function.
        """

        self._assert_attached_before_runtime()

        inputs = inputs or {}
        outputs = outputs or {}

        inputs_np: Dict[Tensor, np.ndarray] = {
            h2d: to_numpy(arr, downcast=downcast_inputs)
            for h2d, arr in inputs.items()
        }

        self._validate_run_inputs(inputs_np)
        self._validate_run_outputs(outputs)

        stepio_inputs: Dict[str, np.ndarray] = {
            h2d.tensor_id: arr
            for h2d, arr in inputs_np.items()
        }

        stepio_outputs: Dict[str, np.ndarray] = {
            d2h.tensor_id: arr
            for d2h, arr in outputs.items()
        }

        stepio = popart.PyStepIO(stepio_inputs, stepio_outputs)
        stepio.enableRuntimeAsserts(False)
        self._pb_session.run(stepio)
        # Arrays in outputs should now alias those filled in by _pb_session.run

        # As we have run a program on device that can invalidate the host weight
        # buffers (make it so they no longer reflect the latest device values),
        # we must mark them as out-of-sync.
        self._pb_session.markHostWeightsOutOfSync()

    def run(
            self,
            inputs: Optional[h2dStreamBufferMaps] = None,
            downcast_inputs: bool = True,
    ) -> d2hStreamBufferMaps:
        """Run :func:`~popxl.Session.run_with_outputs` but create the expected outputs and return them.

        Args:
            inputs (h2dStreamBufferMaps, optional): The inputs to the model. Defaults to None.
            downcast_inputs (bool): If True 64-bit float/ints inputs will be downcast to 32-bit variants. Defaults to True.

        Returns:
            d2hStreamBufferMaps: The map of outputs from the model.

        Raises:
            ValueError: If not attached to device before calling this function.
        """

        self._assert_attached_before_runtime()

        # Get D2HStream -> array outputs, convert to TensorId -> array anchors
        outputs = self.create_host_outputs()

        # Can forward inputs directly to run_with_outputs; it will validate.

        self.run_with_outputs(inputs, outputs, downcast_inputs)

        return outputs

    def weights_to_host(self) -> None:
        """Copy the weights to host from the device.

        Raises:
            ValueError: If not attached to device before calling this function.
        """
        self._assert_attached_before_runtime()
        # NOTE: Copies from device, to internal buffers, to Ir TensorData
        #       buffers. After this call, `get_tensor_data` can immediately
        #       return the Ir tensors' TensorData buffers.
        # NOTE: Internally detects if host weights out-of-sync and marks them as
        #       in-sync.
        self._pb_session.copyDeviceWeightsToHost()

    def weights_from_host(self) -> None:
        """Copy the weights to device from the host.

        Raises:
            ValueError: If not attached to device before calling this function.
        """
        self._assert_attached_before_runtime()
        self._pb_session.weightsFromHost()

    def expected_inputs(self) -> List[HostToDeviceStream]:
        """Return the list of expected inputs for this session.

        Data will need to be provided for each of these when doing :func:`~popxl.Session.run`.

        Returns:
            List[HostToDeviceStream]: A list of all the host to device streams
            required by this session.
        """
        return self.ir.get_all_h2d_streams()

    def get_tensor_data(self, tensor: Union[Variable, Constant]) -> np.ndarray:
        """Get the data stored in the tensor on the device including IPU and remote memory.

        This will sync all the host buffers with the corresponding tensors on the device. Note this
        is a memory view of the data, so will not allocate extra memory for the data, but it is your
        responsibility to ensure the data in the tensor is live at the point of retrieval.

        Args:
            tensor (Union[Variable, Constant]): The tensor to get the data for. Must be Constant or
            Variable type.

        Returns:
            np.ndarray: The data for the tensor in question, with type and shape the same as the
            device tensor.
        """

        return self.get_tensors_data([tensor])[tensor]

    def get_tensors_data(self, tensors: Iterable[Union[Variable, Constant]]
                         ) -> Dict[Union[Constant, Variable], np.ndarray]:
        """Call `get_tensor_data` for multiple tensors.

        This will only sync the host and device buffers once.

        Args:
            tensors (Iterable[Union[Variable, Constant]]): An iterable of the tensors to provide data
            for.

        Raises:
            TypeError: If any tensor is not of type Constant, Variable.

        Returns:
            Dict[Union[Constant, Variable], np.ndarray]: A dictionary of tensors and the
            corresponding data arrays returned.
        """

        # Guard against bad argument.
        any_variable = False
        for obj in tensors:
            if not isinstance(obj, (Constant, Variable)):
                raise TypeError(
                    f"{obj} is not of type Constant or Variable. get_tensor_data is not"
                    "supported for this type.")
            if isinstance(obj, Variable):
                any_variable = True

        # Fetch the latest weights from device into the TensorData buffers.
        # We skip this step if:
        #   1) Only Constants were requested. By definition, these cannot have
        #      been updated on device, so we do not bother to fetch the latest
        #      values.
        #   2) We are not attached to the device. This occurs if and only if
        #      we are outside the Session context. In this case, we can only
        #      return the current host weights.
        #   3) The host weights are already in sync. There is no need to fetch
        #      the weights again.
        if any_variable and self.is_attached and not self._pb_session.areHostWeightsInSync(
        ):
            self.weights_to_host()

        return_tensors: Dict[Union[Constant, Variable], np.ndarray] = {}

        for tensor in tensors:
            if isinstance(tensor, Variable) and (
                    tensor.retrieval_mode == "all_replicas") and (
                        tensor.replica_grouping.group_size > 1):
                # If using all replicas retrieval mode, we must use weightsIo to copy
                # every replica's weights.
                shape = list(tensor.shape_on_host)
                shape[0] *= tensor.replica_grouping.group_size
                weights = {}
                weights[tensor.id] = np.empty(shape, tensor.dtype.as_numpy())
                weightsIo = popart.PyWeightsIO(weights)
                self._pb_session.readWeights(weightsIo)

                return_tensors[tensor] = weights[tensor.id]
            else:
                return_tensors[tensor] = _popxl_to_numpy(tensor)

        return return_tensors

    def write_variable_data(self, tensor: Variable, data: np.ndarray) -> None:
        """Write the variable tensor data from the provided host array.

        This is only valid for Variable type tensors.

        tensor and data must have matching shape and dtype.

        If attached to device, the Variable will be updated on host, then a
        ``weights_from_host`` will occur to update the weights on device.

        If not attached to device, the Variable will be updated on host only.
        The next call to `weights_from_host` will send this updated value to
        device.

        Args:
            tensor (Variable): The popxl tensor to update.
            data (np.ndarray): The array to update the tensor data to.

        Raises:
            TypeError: If the tensor is not a Variable.
            TypeError: If the data types do not match.
            ValueError: If the shapes do not match.
        """

        self.write_variables_data({tensor: data})

    def write_variables_data(self,
                             tensors: Dict[Variable, np.ndarray]) -> None:
        """Call ``write_variable_data`` for multiple tensors in one go.

        Like ``write_variable_data``, the ``weights_from_host`` will only occur
        if already attached to device when calling this function.

        The host to device transfer is delayed until the end so that it only
        happens once.

        Raises:
            TypeError: If any input tensor is not of type Variable.

            TypeError: If the input array data type does not match that of the associated
                tensor.

            ValueError: If the input array shape does not match that of the associated
                tensor.

            NotImplementedError: If the retreval mode of the variable is "all_replicas.
                This is currently not supported.

        Args:
            tensors Dict[(Variable, np.ndarray]): A dictionary of tensors and
            the corresponding array to call 'write_variable_data` with.
        """

        import numbers
        for tensor, data in tensors.items():
            if not isinstance(tensor, Variable):
                raise TypeError(
                    f"Tensor {tensor.id} is not of type Variable. write_variable_data is not"
                    "supported for this type.")

            if isinstance(data, numbers.Number):
                data = np.array(data).astype(tensor.dtype.as_numpy())
            if data.dtype != tensor.dtype.as_numpy():
                raise TypeError(
                    f"The dtype of the input array {data.dtype} must match the equivalent "
                    f"type of the tensor {tensor.id} : {tensor.dtype}")
            elif data.shape != tensor.shape_on_host:
                raise ValueError(
                    f"The shape of the input array {data.shape} must match the "
                    f"shape of the tensor {tensor.id} : {tensor.shape}")
            elif isinstance(tensor, Variable) and (
                    tensor.retrieval_mode == "all_replicas") and (
                        tensor.replica_grouping.group_size > 1):
                raise NotImplementedError(
                    f"Copying to tensor {tensor.id} with \"all_replicas\" "
                    "retreval mode is not yet supported.")

            tensor._pb_tensor.writeTensorData(data, self._pb_session)

        if self.is_attached:
            self.weights_from_host()

    def create_host_outputs(self) -> d2hStreamBufferMaps:
        """
        Return a mapping from popxl.DeviceToHostStream to an empty np.ndarray.

        Later, this can be passed to `session.run_with_outputs`,
        which will fill each array with the values streamed back from device.

        There is an entry in the mapping for every stream in
        `ir.get_all_d2h_streams()`, for the Ir that this Session was
        constructed for.

        For stream s, the shape of the np.ndarray it maps to is:

          `(d, r) + s.shape`

        Where:

          * `d` = `ir.num_host_transfers`
          * `r` = `ir.instance_replication_factor`

        And all dimensions not >1 in `(d, r)` will be removed.

        Examples:

        If:

        .. code-block:: python

          ir.num_host_transfers = 4
          ir.instance_replication_factor = 16
          s.shape = (2, 4)

        Then the shape will be `(4, 16, 2, 4)`

        If:

        .. code-block:: python

          ir.num_host_transfers = 1
          ir.instance_replication_factor = 16
          s.shape = (2, 4)

        Then the shape will be `(16, 2, 4)`

        If:

        .. code-block:: python

          ir.num_host_transfers = 4
          ir.instance_replication_factor = 1
          s.shape = (2, 4)

        Then the shape will be `(4, 2, 4)`

        If:

        .. code-block:: python

          ir.num_host_transfers = 1
          ir.instance_replication_factor = 1
          s.shape = (2, 4)

        Then the shape will be `(2, 4)`

        NOTE: Batch serialisation is not supported, so there is no dimension
        for this.
        """

        outputs = {}

        for s in self._expected_outputs():
            outputs[s] = np.empty(self._full_input_shape(s.shape),
                                  dtype=s.dtype.as_numpy())

        return outputs

    # Properties:
    @property
    def ir(self) -> Ir:
        """
        Return the associated Ir for this session.

        Read only.
        """
        return self.ir_

    @property
    def is_attached(self) -> bool:
        """Return if the session is attached to a device."""
        return self._device.isAttached

    def _set_device(self, device: popart.DeviceInfo):
        """
        Change the Session to use a different device.

        Args:
            device (popart.DeviceInfo): Device to use.
        """
        self._device = device
        self._pb_session._setDeviceInfo(device)

    @contextmanager
    def _cleanup_on_error(self):
        """Call `Session.__exit__` if an exception in `Session.__enter__` is raised.

        Function calls which can raise unhandled exceptions in `Session.__enter__`
        should use this context manager.

        This function was inspired by the recipes in:
        https://docs.python.org/3/library/contextlib.html#cleaning-up-in-an-enter-implementation
        """
        with ExitStack() as stack:
            stack.push(self)
            yield
            stack.pop_all()

    def __enter__(self) -> 'Session':
        """
        Enter the context of this ``Session``.

        If not already attached to a device, this will attach to an available device and perform a
        ``weights_from_host``.

        See :numref:`sec_session` for a more comprehensive
        guide to sessions and the context manager.
        """
        # Only weights_from_host if going from detached->attached.
        should_setup = not self.is_attached

        # If the current device is an OfflineIpu then acquire a proper device
        if self._device.type == popart.DeviceType.OfflineIpu:
            # Store the offline device to be used on context exit
            self._was_attached_stack.append(self._device)
            self._set_device(
                _acquire_hw_device_with_timeout(self._device.numIpus))
        else:
            self._was_attached_stack.append(self.is_attached)
            # This is needed for when device has been provided by the user.
            if not self.is_attached:
                # Attach methods do not block, instead they return False if failed
                self._device.attach()
                if not self.is_attached:
                    # This will wait for OnDemandAttachTimeout if set on the associated device manager.
                    self._device.tryAttachUntilTimeout()
                if not self.is_attached:
                    raise RuntimeError(
                        f"Could not attach to device {self._device.id}. Check `gc-monitor` for device usage."
                    )

        if should_setup:
            with self._cleanup_on_error():
                self.weights_from_host()

        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        """
        Exit the context of this ``Session``.

        If you were not attached when entering this context, this will perform a
        ``weights_to_host`` then detach.

        See :numref:`sec_session` for a more comprehensive
        guide to sessions and the context manager.
        """

        # Teardown context only if we were not attached when entering it. Thus
        # if we were already attached on enter, we will still be attached after
        # exit.
        was_attached_or_device = self._was_attached_stack.pop()

        should_teardown = not was_attached_or_device or isinstance(
            was_attached_or_device, popart.DeviceInfo)
        if should_teardown:
            if exc_type is None:
                self.weights_to_host()
            self._device.detach()
            self._pb_session.setEngineIsLoaded(False)

        # If a DeviceInfo was stored in the stack then restore it.
        if isinstance(was_attached_or_device, popart.DeviceInfo):
            self._set_device(was_attached_or_device)

        # Note, if the user was attached, but then manually detached inside the
        # context, we do not restore their pre-enter attached state.

        # Let exceptions propagate

    # Private methods
    def _get_ipus_per_replica(self) -> int:
        ir_ipus = set(ipu for g in self._ir.getAllGraphs()
                      for ipu in g.getAllVirtualGraphIds(True))
        if not ir_ipus:
            raise RuntimeError(
                f"The Ir {self.ir.id} has no graphs. The graphs may have all been optimised to"
                "nothing, try adding more operations to your graphs.")
        return max(ir_ipus) + 1

    def _get_ipu_count(self) -> int:
        """Return the number of ipus required by this session and ir.

        Equal to 2 ** ceil(log_2(max(virtual_graphs) + 1))

        Raises:
            RuntimeError: If the Ir has no graphs.

        Returns:
            int: The number of ipus required by this session + ir.
        """
        num_ipus = self._get_ipus_per_replica()
        num_ipus *= self.ir.instance_replication_factor
        return 2**math.ceil(math.log2(num_ipus))

    def _expected_outputs(self) -> List[DeviceToHostStream]:
        """Return the list of expected outputs from this session.

        Data will be returned for each of these when doing `:func:`~popxl.Session.run``.

        Returns:
            List[DeviceToHostStream]: A list of all the device to host streams
            returned by this session.
        """
        return self.ir.get_all_d2h_streams()

    def _full_input_shape(self, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Return the full input shape that this array will need to be.

        The shape is taking into account num_host_transfers and replication_factor.

        For example, shape = (3, 4), num_host_transfers = 8, replicas = 4, _full_input_shape =
        (8, 4) + (3, 4)  (8, 4, 3, 4)

        Args:
            shape (Tuple[int, ...]): The shape to add the additional dims to.

        Returns:
            Tuple[int, ...]: The full input shape for this array shape.
        """
        return self._extra_input_dims + shape

    def _validate_run_io_streams(self, stream_buffer_map: StreamBufferMaps,
                                 expected_streams: StreamBufferMaps) -> None:
        """Validate that the from / to device streams are present and valid.

        Checks there are no missing or unexpected streams, then checks the shape of each is correct.

        Args:
            stream_buffer_map (StreamBufferMaps): The streams provided by the user.
            expected_streams (StreamBufferMaps): The expected streams for the session.

        Raises:
            ValueError: If There are missing or unexpected streams.
        """
        # 1: Validate no missing or unexpected streams.

        set_streams = set(stream_buffer_map.keys())
        set_expected_streams = set(expected_streams)

        stream_type_str = ""

        for s in set_streams:
            stream_type_str = "input" if isinstance(
                s, HostToDeviceStream) else "output"
            break

        if set_streams != set_expected_streams:

            unexpected = {str(s) for s in set_streams - set_expected_streams}
            missing = {str(s) for s in set_expected_streams - set_streams}
            raise ValueError(f"Unexpected/Missing {stream_type_str}.\n  "
                             f"Unexpected: {unexpected}\n  Missing: {missing}")

        # 2: Validate arrays have correct shape for streams.
        for s, arr in stream_buffer_map.items():
            self._verify_io_shape(s, arr)

    def _verify_io_shape(self,
                         s: Union[DeviceToHostStream, HostToDeviceStream],
                         arr: np.ndarray) -> None:
        """Verify the array shape is as expected for a DeviceToHostStream or HostToDeviceStream.

        Args:
            s (Union[DeviceToHostStream, HostToDeviceStream]): The stream to check.
            arr (np.ndarray): The corresponding array to check against the stream.

        Raises:
            ValueError: If the num_host_transfers dimension of the array is != num_host_transfers
            ValueError: If the replication dimension of the array != replication_factor
            ValueError: If the remaining dimensions are not equal to the stream shape.
        """
        stream_type_str = "input" if isinstance(
            s, HostToDeviceStream) else "output"
        full_shape = self._full_input_shape(s.shape)
        # Data is always split for num_host_transfers at index 0 (if enabled).
        # Index at which the data is split for replication.
        repl_index = 0
        # Index at which the data dimensions start.
        data_index = 0
        if self.ir.num_host_transfers > 1:
            repl_index += 1
            data_index += 1
            if arr.shape[0] < self.ir.num_host_transfers:
                raise ValueError(
                    f"Dimension 0 ({arr.shape[0]}) for the array provided for "
                    f"{stream_type_str} stream {s.tensor_id} is not large enough.\n"
                    f"It should be at least of size num_host_transfers = {self.ir.num_host_transfers}"
                )
        if self.ir.instance_replication_factor > 1:
            data_index += 1
            if arr.shape[repl_index] != self.ir.instance_replication_factor:
                raise ValueError(
                    f"Dimension {repl_index} ({arr.shape[repl_index]}) for the array provided for {stream_type_str} "
                    f"stream {s.tensor_id} is the wrong size.\n"
                    f"It should be of size replication_factor = {self.ir.instance_replication_factor}"
                )
        if arr.squeeze().shape[data_index:] == ():
            # special case, difficult to compare dimensions on arrays with dimensions all 1's
            return
        if arr.shape[data_index:] != s.shape:
            raise ValueError(
                f"Shape mismatch for {stream_type_str} stream {s.tensor_id}:\n"
                f"Stream shape = {s.shape}, therefore expected full global batch shape "
                f"{full_shape}. Got array shape = {arr.shape}")

    def _validate_run_inputs(self, inputs: h2dStreamBufferMaps) -> None:
        """Run :func:`~popxl.Session._validate_run_io_streams` for each of the given inputs vs.
        the expected inputs.

        Args:
            inputs (h2dStreamBufferMaps): The inputs to check.
        """
        self._validate_run_io_streams(inputs, self.expected_inputs())

    def _validate_run_outputs(self, outputs: d2hStreamBufferMaps) -> None:
        """Run :func:`~popxl.Session._validate_run_io_streams` for each of the given outputs vs.
        the expected outputs.

        Args:
            outputs (h2dStreamBufferMaps): The outputs to check.
        """
        self._validate_run_io_streams(outputs, self._expected_outputs())

    # Private properties:
    @property
    def _ir(self) -> _ir.Ir:
        """
        Return the associated popart._internal.ir Ir object for this session.

        Read only.
        """
        return self.ir._pb_ir

    @property
    def _extra_input_dims(self) -> Tuple[int, ...]:
        """Return the tuple of extra input dimensions required for this session.

        Equal to (num_device_iterations, num_replicas)

        Returns:
            Tuple[int, ...]: A tuple (num_device_iterations, num_replicas)
        """
        _extra_input_dims: Tuple[int, ...] = tuple()
        if self.ir.num_host_transfers > 1:
            _extra_input_dims += (self.ir.num_host_transfers, )
        if self.ir.instance_replication_factor > 1:
            _extra_input_dims += (self.ir.instance_replication_factor, )
        return _extra_input_dims
