# Copyright (c) 2022 Graphcore Ltd. All rights reserved.
import math
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import popart
import popart._internal.ir as _ir
from popart.ir.ir import Ir
from popart.ir.streams import DeviceToHostStream, HostToDeviceStream
from popart.ir.tensor import Constant, HostTensor, Variable
from typing_extensions import Literal

from popart.ir.utils import _to_device_info, _to_numpy

d2hStreamBufferMaps = Union[Mapping[DeviceToHostStream, np.ndarray],
                            Mapping[DeviceToHostStream, HostTensor]]
h2dStreamBufferMaps = Union[Mapping[HostToDeviceStream, np.ndarray],
                            Mapping[HostToDeviceStream, HostTensor]]
StreamBufferMaps = Union[h2dStreamBufferMaps, d2hStreamBufferMaps]


class Session:
    """Class that represents the PopART runtime session.

    A class that allows you to execute a `popart.ir` graph of operations.

    .. warning:: The session object takes ownership of the provided Ir and it cannot be modified afterwards.
    """

    def __init__(self,
                 ir: Ir,
                 device_desc: Literal["ipu_hw", "ipu_model", "cpu"] = "cpu"
                 ) -> None:
        """Initialise a new session.

        Args:
            ir (Ir): The Ir to use for this session.
            device_desc (Literal["ipu_hw", "ipu_model", "cpu"], optional): The type of
                ipu device to use. One of:
                "ipu_hw": Real IPU hardware. Uses `DeviceConnectionType` == `OnDemand` and
                    `DeviceSelectionCriterion` == `Random`.
                "ipu_model": IPU model.
                "cpu": CPU model. Does not support replication.
                Defaults to "ipu_model".
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

        num_ipus = self._get_ipu_count()
        device = None
        device = _to_device_info(device_type=device_desc, num_ipus=num_ipus)
        if device is None:
            raise RuntimeError(
                f"Could not aquire a device with device_type={device_desc}, \
            num_ipus={num_ipus}.")

        self._device = device

        # Note: This uses the underlying popart .InferenceSession class, but supports BOTH inference
        # and training (via the autodiff transform). We use the  popart.InferenceSession class to
        # avoid any automatic autodiff-like transformations we want to do manually in popart.ir.
        self._pb_session = popart.InferenceSession.fromIr(ir=self._ir,
                                                          deviceInfo=device)

        self._pb_session.prepareDevice()
        self._pb_session.weightsFromHost()

    def _get_ipu_count(self) -> int:
        """Returns the number of ipus required by this session and ir.

        Equal to 2 ** ceil(log_2(max(virtual_graphs) + 1))

        Returns:
            int: The number of ipus required by this session + ir.
        """
        ir_ipus = set(ipu for g in self._ir.getAllGraphs()
                      for ipu in g.getAllVirtualGraphIds(True))
        if ir_ipus:
            num_ipus = max(ir_ipus) + 1
        else:
            # Edge case : ir_ipus = {}, no graphs found, this leads to an incomprehensible error.
            raise RuntimeError(
                f"The Ir {self.ir.id} has no graphs. The graphs may have all been optimised to"
                "nothing, try adding more operations to your graphs.")
        if self._ir.getSessionOptions().enableReplicatedGraphs:
            num_ipus *= self._ir.getSessionOptions().replicatedGraphCount
        return 2**math.ceil(math.log2(num_ipus))

    def weights_to_host(self) -> None:
        """Copy the weights to host from the device."""
        self._pb_session.weightsToHost()

    def weights_from_host(self) -> None:
        """Copy the weights to device from the host."""
        self._pb_session.weightsFromHost()

    def expected_inputs(self) -> List[HostToDeviceStream]:
        """Returns the list of expected inputs for this session.

        Data will need to be provided for each of these when doing `:func:`~pir.Session.run``.

        Returns:
            List[HostToDeviceStream]: A list of all the host to device streams
            required by this session.
        """
        return self.ir.get_all_h2d_streams()

    def _expected_outputs(self) -> List[DeviceToHostStream]:
        """Returns the list of expected outputs from this session.

        Data will be returned for each of these when doing `:func:`~pir.Session.run``.

        Returns:
            List[DeviceToHostStream]: A list of all the device to host streams
            returned by this session.
        """
        return self.ir.get_all_d2h_streams()

    def _full_input_shape(self, shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Returns the full input shape that this array will need to be, when taking into account
        num_host_transfers and replication_factor.

        For example, shape = (3, 4), device_iterations = 8, replicas = 4, _full_input_shape =
        (8, 4) + (3, 4)  (8, 4, 3, 4)

        Args:
            shape (Tuple[int, ...]): The shape to add the additional dims to.

        Returns:
            Tuple[int, ...]: The full input shape for this array shape.
        """
        return self._extra_input_dims + shape

    def get_tensor_data(self, tensor: Union[Variable, Constant]) -> np.ndarray:
        """Get the data stored in the tensor on the device.

        This will sync all the host buffers with the corresponding tensors on the device.

        Args:
            tensor (Union[Variable, Constant]): The tensor to get the data for. Must be Constant or
            Variable type.

        Raises:
            TypeError: If the tensor is not of type Constant, Variable.

        Returns:
            np.ndarray: The data for the tensor in question, with type and shape the same as the
            device tensor.
        """
        if not isinstance(tensor, (Constant, Variable)):
            raise TypeError(
                f"Tensor {tensor.id} is not of type Constant or Variable. get_tensor_data is not"
                "supported for this type.")
        if isinstance(tensor, Variable):
            self._pb_session.copyToTensorData()

        return _to_numpy(tensor)

    def get_tensors_data(self, tensors: Iterable[Union[Variable, Constant]]
                         ) -> Dict[Union[Constant, Variable], np.ndarray]:
        """Call `get_tensor_data` for multiple tensors.

        This will only sync the host and device buffers once.

        Args:
            tensors (Iterable[Union[Variable, Constant]]): An iterable of the tensors to provide data
            for.

        Returns:
            Dict[Union[Constant, Variable], np.ndarray]: A dictionary of tensors and the
            corresponding data arrays returned.
        """
        return_tensors: Dict[Union[Constant, Variable], np.ndarray] = {}
        self._pb_session.copyToTensorData()
        for tensor in tensors:
            return_tensors[tensor] = _to_numpy(tensor)

        return return_tensors

    def write_variable_data(self,
                            tensor: Variable,
                            data: np.ndarray,
                            write_from_host: bool = True) -> None:
        """Writes the variable tensor data from the provided host array.

        This is only valid for Variable type tensors. This will update values on the device with
        values from the array, both the Tensor and the array must have matching types and shapes.

        Args:
            tensor (Variable): The tensor to update on the device.
            data (np.ndarray): The array with values to update to.
            write_from_host (bool, optional): Whether to do the actual transfer to device; used for
                delaying multiple transfers to run in one go, for example in ``write_tensors_data``.
                Defaults to True.

        Raises:
            TypeError: If the tensor is not a Variable.
            TypeError: If the data types do not match.
            ValueError: If the shapes do not match.
        """
        import numbers
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
        elif data.shape != tensor.shape:
            raise ValueError(
                f"The shape of the input array {data.shape} must match the "
                f"shape of the tensor {tensor.id} : {tensor.shape}")

        tensor._pb_tensor.writeTensorData(data)
        if write_from_host:
            self.weights_from_host()

    def write_variables_data(self,
                             tensors: Dict[Variable, np.ndarray]) -> None:
        """Calls ``write_variable_data`` for multiple tensors, but delays the host to device transfer
        until last so that it is transferred in one go.

        Args:
            tensors (Variable, np.ndarray]): A dictionary of tensors and the
            corresponding array to call 'write_variable_data` with.
        """
        for k, v in tensors.items():
            self.write_variable_data(k, v, False)
        self.weights_from_host()

    def create_host_outputs(self) -> d2hStreamBufferMaps:
        """
        Returns a mapping from popart.ir.DeviceToHostStream to an empty
        np.ndarray. Later, this can be passed to `session.run_with_outputs`,
        which will fill each array with the values streamed back from device.

        There is an entry in the mapping for every stream in
        `ir.get_all_d2h_streams()`, for the Ir that this Session was
        constructed for.

        For stream s, the shape of the np.ndarray it maps to is:
          `(d, r) + s.shape`
        where
          * `d` = `ir.num_host_transfers`
          * `r` = `ir.replication_factor`
        and all dimensions not >1 in `(d, r)` will be removed.

        Examples:

        If:
        .. code-block:: python

          ir.num_host_transfers = 4
          ir.replication_factor = 16
          s.shape = (2, 4)

        Then the shape will be `(4, 16, 2, 4)`

        If:
        .. code-block:: python

          ir.num_host_transfers = 1
          ir.replication_factor = 16
          s.shape = (2, 4)

        Then the shape will be `(16, 2, 4)`

        If:
        .. code-block:: python

          ir.num_host_transfers = 4
          ir.replication_factor = 1
          s.shape = (2, 4)

        Then the shape will be `(4, 2, 4)`

        If:
        .. code-block:: python

          ir.num_host_transfers = 1
          ir.replication_factor = 1
          s.shape = (2, 4)

        Then the shape will be `(2, 4)`

        NOTE: Batch serialisation is not yet supported, so there is no dimension
        for this yet.
        """

        outputs = {}

        for s in self._expected_outputs():
            outputs[s] = np.empty(self._full_input_shape(s.shape),
                                  dtype=s.dtype.as_numpy())

        return outputs

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
                s, DeviceToHostStream) else "output"
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
                         arr: HostTensor) -> None:
        """Verify the array shape is as expected for a DeviceToHostStream or HostToDeviceStream.

        Args:
            s (Union[DeviceToHostStream, HostToDeviceStream]): The stream to check.
            arr (HostTensor): The corresponding array to check against the stream.

        Raises:
            ValueError: If the num_host_transfers dimension of the array is != num_host_transfers
            ValueError: If the replication dimension of the array != replication_factor
            ValueError: If the remaining dimensions are not equal to the stream shape.
        """
        stream_type_str = "input" if isinstance(
            s, DeviceToHostStream) else "output"
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
        if self.ir.replication_factor > 1:
            data_index += 1
            if arr.shape[repl_index] != self.ir.replication_factor:
                raise ValueError(
                    f"Dimension {repl_index} ({arr.shape[1]}) for the array provided for {stream_type_str} "
                    f"stream {s.tensor_id} is the wrong size.\n"
                    f"It should be of size replication_factor = {self.ir.replication_factor}"
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
        """Run :func:`~pir.Session._validate_run_io_streams` for each of the given inputs vs.
        the expected inputs.

        Args:
            inputs (h2dStreamBufferMaps): The inputs to check.
        """
        self._validate_run_io_streams(inputs, self.expected_inputs())

    def _validate_run_outputs(self, outputs: d2hStreamBufferMaps) -> None:
        """Run :func:`~pir.Session._validate_run_io_streams` for each of the given outputs vs.
        the expected outputs.

        Args:
            outputs (h2dStreamBufferMaps): The outputs to check.
        """
        self._validate_run_io_streams(outputs, self._expected_outputs())

    def run_with_outputs(
            self,
            inputs: Optional[h2dStreamBufferMaps] = None,
            outputs: Optional[d2hStreamBufferMaps] = None) -> None:
        """Run this session with the provided inputs and outputs. Inputs will be used as inputs to
        the model, and outputs will be written to by the session.

        Args:
            inputs (h2dStreamBufferMaps, optional): The inputs to the model. Defaults to None.
            outputs (d2hStreamBufferMaps, optional): The output buffers, these will be written to
                and modified. Defaults to None.
        """

        inputs = inputs or {}
        outputs = outputs or {}

        self._validate_run_inputs(inputs)
        self._validate_run_outputs(outputs)

        stepio_inputs: Mapping[str, np.ndarray] = {
            h2d.tensor_id: arr
            for h2d, arr in inputs.items()
        }

        stepio_outputs: Dict[str, np.ndarray] = {
            d2h.tensor_id: arr
            for d2h, arr in outputs.items()
        }

        stepio = popart.PyStepIO(stepio_inputs, stepio_outputs)
        stepio.enableRuntimeAsserts(False)
        self._pb_session.run(stepio)

        # Arrays in outputs should now alias those filled in by _pb_session.run

    def run(self, inputs: Optional[h2dStreamBufferMaps] = None
            ) -> d2hStreamBufferMaps:
        """Run :func:`~pir.Session.run_with_outputs` but create the expected outputs and return them.

        Args:
            inputs (h2dStreamBufferMaps, optional): The inputs to the model. Defaults to None.

        Returns:
            d2hStreamBufferMaps: The map of outputs from the model.
        """
        # Get D2HStream -> array outputs, convert to TensorId -> array anchors
        outputs: Mapping[DeviceToHostStream, np.
                         ndarray] = self.create_host_outputs()

        # Can forward inputs directly to run_with_outputs; it will validate.

        self.run_with_outputs(inputs, outputs)

        return outputs

    # Properties:
    @property
    def _extra_input_dims(self) -> Tuple[int, ...]:
        """The tuple of extra input dimensions required for this session. Equal to
        (num_device_iterations, num_replicas)

        Returns:
            Tuple[int, ...]: A tuple (num_device_iterations, num_replicas)
        """
        _extra_input_dims: Tuple[int, ...] = tuple()
        if self.ir.num_host_transfers > 1:
            _extra_input_dims += (self.ir.num_host_transfers, )
        if self.ir.replication_factor > 1:
            _extra_input_dims += (self.ir.replication_factor, )
        return _extra_input_dims

    @property
    def ir(self) -> Ir:
        """The associated Ir for this session. Read only.
        """
        return self.ir_

    @property
    def _ir(self) -> _ir.Ir:
        """The associated popart._internal.ir Ir object for this session. Read only.
        """
        return self.ir._pb_ir

    @property
    def device(self) -> popart.DeviceInfo:
        """The popart.DeviceInfo object representing the device for this session to run on.
        """
        return self._device

    @device.setter
    def device(self, device: popart.DeviceInfo) -> None:
        """Setter for :func:`~pir.Session.device`

        Args:
            device (popart.DeviceInfo): The popart.DeviceInfo to set this to.
        """
        self._device = device
