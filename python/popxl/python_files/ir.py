# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
"""Definition of a class that represents the PopART IR."""
import inspect
import os
from collections import OrderedDict
from pathlib import Path
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterable, List,
                    Optional, Set, Tuple, Union)
from typing_extensions import Literal
from weakref import WeakValueDictionary

import popart
import popart._internal.ir as _ir
from popxl.graph import Graph
from popxl.module import Module
from popxl.tensor import Tensor, TensorByRef, TensorSpec, graph_input, graph_output
from popxl.replica_grouping import ReplicaGrouping

if TYPE_CHECKING:
    IrCache = WeakValueDictionary[int, 'Ir']


class Ir:
    def __init__(self, replication: Union[int, Literal['popdist']] = 1):
        """
        PopXL intermediate representation (IR).

        An IR contains a main graph (property `main_graph`) and can create
        additional graphs using member methods such as :py:meth:`create_graph` and
        :py:meth:`create_empty_graph`.

        Args:
            replication (Union[int, Literal['popdist']], optional):
                Set the replication_factor of the IR. Value of 'popdist' configures the IR with settings from popdist/poprun. Defaults to 1.
        """
        self._pb_ir = _ir.Ir()
        # Set better defaults for popxl programs.
        # Some parts of graph construction use the session options to make decisions,
        # such as inheritPlacementAttributes and scheduling priorities. So we must set
        # the options when we construct the _internal IR before the user starts constructing
        # the graphs.
        opts = self._pb_ir.getSessionOptions()
        opts.virtualGraphMode = popart.VirtualGraphMode.Manual
        opts.useHostCopyOps = True
        opts.aliasZeroCopy = True
        opts.enableExplicitMainLoops = True
        opts.explicitRecomputation = True
        opts.enableInplaceAmbiguityChecking = True

        self._use_popdist = replication == 'popdist'
        if self._use_popdist:
            import popdist.popart
            popdist.popart.configureSessionOptions(opts)
        else:
            self.replication_factor = replication

        Ir._ir_cache[self.id] = self
        self._graph_cache: Dict[str, Graph] = {}
        # If no num_device_iterations, default to 1
        if not self._pb_ir.getDataFlow().batchesPerStep():
            self._pb_ir.getDataFlow().setBatchesPerStep(1)

    _ir_cache: 'IrCache' = WeakValueDictionary()

    @classmethod
    def _from_pb(
            cls,
            pb_ir: '_ir.Ir',
    ) -> 'Ir':
        """Return popxl.Ir for a pybind _ir.Ir.

        Used as a factory method.

        Args:
            pb_ir (_ir.Ir):
                An instance of the low-level pybind11 `Ir`.

        Raises:
            RuntimeError: If the pb_ir does not already have a popxl Ir in-memory.

        Returns:
            Ir:
            A popxl.Ir that represents the passed pb_ir.
        """
        _id = pb_ir.getId()
        if _id not in Ir._ir_cache:
            raise RuntimeError(
                "Constructing a new Ir with _from_pb is unexpected. "
                "This implies the Ir was garbage collected. "
                "Is a popxl class missing a back reference to Ir?")
        return Ir._ir_cache[_id]

    @property
    def main_graph(self) -> 'Graph':
        """
        Every IR is initialised with a main graph. This method returns this graph.

        Returns:
            Graph:
                The main graph of the IR.
        """
        return Graph._from_pb(self._pb_ir.getMainGraph())

    def create_graph(
            self,
            fn: Union[Callable[..., Union[None, Tensor, Iterable[Tensor]]],
                      Module],
            *args: Any,
            **kwargs: Any,
    ) -> 'Graph':
        """
        Create a graph from a Python callable `fn` or the build method of a `Module`.
        The graph inputs are determined using the signature of the function `fn`
        and the supplied arguments `args` and `kwargs`. Tensors or TensorSpecs passed via the
        arguments are used to determine the shape and dtype of the graph inputs (the
        tensors are not actually passed to the graph). The graph outputs are
        determined using the outputs of the function when called.

        The order of inputs in the returned graph will be the same as the
        order of the tensor inputs in the function signature, the
        order of the kwargs and the order of called `popxl.graph_inputs`.
        This determines the order in which you pass the parent
        tensors as inputs at the callsite.

        The function `fn` can take any arguments. Any Tensor arguments are
        automatically detected. Any Tensor arguments inside a tuple, list,
        `*arg` or `**kwargs` are also detected. `*args`, `**kwargs`, lists
        cannot contain a mixture of tensors and other types. Nested lists
        or dicts of tensors are not supported.

        If an input is type hinted with `TensorByRef` or `List[TensorByRef]`
        where appropriate in the signature of `fn` then the corresponding inputs
        will be passed by reference instead of by value when the graph is called.

        The output of `fn` must be either None, a Tensor or an iterable of Tensors.

        Args:
            fn (Callable[..., Any]):
                The Python function that defines the graph. The signature of
                `fn` with its arguments is used to determine the
                inputs of the graph.
            *args (Any):
                Arguments passed to the Python function that defines the graph
                that can be a mixture of tensors and other types. Tensors are
                used to determine the tensor info of the inputs.
            **kwargs (Any):
                Keyword arguments passed to the Python function that defines the
                graph that can be a mixture of tensors and other types. Tensors
                are used to determine the tensor info of the inputs.

        Raises:
            TypeError: If `fn` is not a callable extending the popxl.Module or if any of the
                arguments listed in *args mixes Tensors with other types
            ValueError: If the *args and **kwargs don't match the signature or if the output
                of a subgraph is not a Tensor, an iterable of Tensors or None.

        Returns:
            Graph:
                A graph that corresponds to the input Python function.
        """
        if isinstance(fn, Module):
            qualname = fn.__class__.__qualname__
            func = fn.build
        else:
            # Note all Python functions will have __qualname__.
            if not callable(fn) or not hasattr(fn, '__qualname__'):
                raise TypeError(
                    "Callable `fn` must be either a function or a class that "
                    "extends popxl.Module")
            else:
                qualname = fn.__qualname__
                func = fn

        signature = inspect.signature(func, follow_wrapped=True)
        try:
            bound_args = signature.bind(*args, **kwargs)
        except TypeError as e:
            raise ValueError(
                "The arguments, `args` and `kwargs`, do not match the signature"
                " of the function `fn`.") from e
        bound_args.apply_defaults()
        arguments = bound_args.arguments

        name = self._create_name(qualname)
        _pb_subgraph = self._pb_ir.createGraph(name)
        subgraph = Graph._from_pb(_pb_subgraph)

        with subgraph:
            for name, arg in bound_args.arguments.items():
                type_hint = signature.parameters[name].annotation

                if isinstance(arg, (Tensor, TensorSpec)):
                    by_ref = type_hint is TensorByRef
                    arguments[name] = graph_input(arg.shape,
                                                  arg.dtype,
                                                  name,
                                                  by_ref=by_ref,
                                                  meta_shape=arg.meta_shape)

                # Supported args:
                # 1. Argument that is a list `def func(x: List[Tensor])`
                # 2. Variable length argument `def func(*x)`
                # 3. Variable keyword arguments `def func(**x)`
                elif isinstance(arg, (tuple, list, dict)):
                    by_ref = False
                    signature_kind = signature.parameters[name].kind

                    if (signature_kind is inspect.Parameter.VAR_POSITIONAL or
                            signature_kind is inspect.Parameter.VAR_KEYWORD):
                        # Variable length argument (*arg)
                        # or variable keyword argument (**kwarg)
                        by_ref = type_hint is TensorByRef
                    elif isinstance(arg, (tuple, list)):
                        # Argument that is a list
                        by_ref = type_hint is List[
                            TensorByRef] or type_hint is Tuple[TensorByRef]
                    elif isinstance(arg, dict):
                        # Argument that is a dict
                        continue

                    if isinstance(arg, dict):
                        items = arg.items()
                    else:
                        items = zip([name + f'_{i}' for i in range(len(arg))],
                                    arg)

                    in_args_sub = OrderedDict()
                    contains_tensor = False

                    for i, (subarg_name, subarg) in enumerate(items):
                        if i > 0 and (isinstance(subarg, (Tensor, TensorSpec))
                                      != contains_tensor):
                            raise TypeError(
                                f"A {type(arg)} argument can't contain a "
                                f"mixture of Tensors and other types. Arg name:"
                                f" {name}. Value: {arg}")
                        if isinstance(subarg, (Tensor, TensorSpec)):
                            contains_tensor = True
                            in_args_sub[subarg_name] = graph_input(
                                subarg.shape,
                                subarg.dtype,
                                subarg_name,
                                by_ref=by_ref,
                                meta_shape=subarg.meta_shape)
                    if contains_tensor and isinstance(arg, dict):
                        arguments[name] = in_args_sub
                    elif contains_tensor and isinstance(arg, (tuple, list)):
                        arguments[name] = list(in_args_sub.values())

            bounds_args_new = inspect.BoundArguments(signature, arguments)
            outputs = fn(*bounds_args_new.args, **bounds_args_new.kwargs)

            if not (outputs is None or isinstance(outputs, Tensor) or
                    (isinstance(outputs, Iterable)
                     and all(isinstance(e, Tensor) for e in outputs))):
                raise ValueError(
                    "Output of subgraph must be None, a Tensor or an iterable of Tensors."
                    f" Output type: {type(outputs)}. Value {outputs}")

            if outputs is None:
                outputs = []

            if isinstance(outputs, Tensor):
                outputs = [outputs]

            for out in outputs:
                graph_output(out)

        return subgraph

    def create_empty_graph(self, name: Optional[str] = None) -> 'Graph':
        """Create a new graph.

        Args:
            name (Optional[str]): Name of the graph. Defaults to "graph".

        Returns:
            Graph: An empty graph.
        """
        name = self._create_name(name or "graph")
        _pb_subgraph = self._pb_ir.createGraph(
            name)  # type: ignore GraphId != str
        return Graph._from_pb(_pb_subgraph)

    def dot_checkpoint(self,
                       check: str,
                       save_dir: Optional[Union[Path, str]] = None) -> None:
        """Output a graphical representation of the graph in Graphviz DOT format.

        Checkpoints can be activated by either setting the `dotChecks` option in session
        options or the `POPART_DOT_CHECKS` environmental variable. These should be set to
        the list of the checks to be activated.
        Note that if either `dotChecks` or `POPART_DOT_CHECKS` is set to `ALL`, all checkpoints
        will be activated.
        See the `PopART User Guide <https://docs.graphcore.ai/projects/popart-user-guide/en/latest/env_vars.html#generating-dot-files>`__ for more information.

        If no checkpoints are activated, this function will activate them all by setting the `dotChecks` option to `ALL`.

        Args:
            check (str): Name of this checkpoint.
            save_dir (Optional[Union[Path, str]]): Directory to store the dot files in.
              Note that this will set the save directory for all dot checkpoints in the graph.
        """
        opts = self._pb_ir.getSessionOptions()
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            opts.logDir = str(save_dir)

        if "POPART_DOT_CHECKS" not in os.environ:
            if len(opts.dotChecks) == 0:
                opts.dotChecks = {
                    "ALL",
                }

        self._pb_ir.dotCheckpoint(self._pb_ir, check)

    def _create_name(self, name: str) -> str:
        """Generate a graph name based on the qualified name of the Python function that created it.

        Note: occurrences of ".<locals>" in the name are removed.

        Example:
            Suppose a graph function:
                >>> class Foo:
                ...     def bar():
                ...         # Graph definition...
            Creating the following graphs:
                >>> ir.create_graph(Foo.bar)
                >>> ir.create_graph(Foo.bar)
            will result in graph names `Foo.bar_0` and `Foo.bar_1`.

        Args:
            name (str):
                The `__qualname__` attribute of the Python function.

        Returns:
            str:
                The name of the graph.
        """
        name = name.replace(".<locals>", "")
        name = self._pb_ir.createUniqueSubgraphId(name)
        return name

    def get_all_d2h_streams(self) -> Set['DeviceToHostStream']:
        """
        Return all ``DeviceToHostStream`` in the IR which has a host_store op that streams along it.
        """
        from popxl.streams import DeviceToHostStream

        # getHostStoreTensors() returns TensorId -> List[Tensor] map. We get the
        # Tensor from the TensorId, convert to popxl.Tensor, then make a
        # popxl.DeviceToHostStream from that.
        return {
            DeviceToHostStream._from_tensor(
                Tensor._from_pb_tensor(
                    self._pb_ir.getTensor(stream_tensor_id)))
            for stream_tensor_id, _host_stored_tensors in
            self._pb_ir.getHostStoreTensors().items()
        }

    def get_all_h2d_streams(self) -> Set['HostToDeviceStream']:
        """
        Return all ``HostToDeviceStream``s in the IR which has a host_load op that streams along it.
        """
        from popxl.streams import HostToDeviceStream

        # getHostLoadTensors() returns TensorId -> List[Tensor] map. We get the
        # Tensor from the TensorId, convert to popxl.Tensor, then make a
        # popxl.HostToDeviceStream from that.
        return {
            HostToDeviceStream._from_tensor(
                Tensor._from_pb_tensor(
                    self._pb_ir.getTensor(stream_tensor_id)))
            for stream_tensor_id, _host_load_tensors in
            self._pb_ir.getHostLoadTensors().items()
        }

    @property
    def num_host_transfers(self) -> int:
        """
        Return the number of fwd-bwd iterations of the model that your Ir computes.

        This property MUST be set before creating a `popxl.Session`.

        More concretely, if your Ir contains an input tensor `x` with shape
        (2, 5), and you expect that your Ir will stream this tensor a total of
        4 times, and therefore you need to pass a buffer with shape (4, 2, 5) to
        each `session.run()` call; then ir.num_host_transfers should equal 4. Note
        there will also be a replica dimension if using replication.

        Note there are no separate values for "batches per step" and "gradient
        accumulation", as they are known in PopART's ONNX API. If your Ir
        represents a batches per step of `bps` and a gradient accumulation
        factor of `af`, then you should set num_host_transfers to `bps * af`.
        There are no separate setters for the two values. There will only be a
        single "num_host_transfers" dimension in the buffer passed to
        `session.run`.
        """
        return self._pb_ir.getDataFlow().batchesPerStep()

    @num_host_transfers.setter
    def num_host_transfers(self, value: int) -> None:
        self._pb_ir.getDataFlow().setBatchesPerStep(value)

    @property
    def replication_factor(self) -> int:
        """Set the number of model replications.

        For example, if your model requires 1 IPU, a `replication_factor` of 2 will replicate your
        model so that 2 IPUs are used. If your model is pipelined across 4 IPUs, a `replication_factor`
        of 4 will use 16 IPUs total. If the training is done across multiple instances then the
        `replication_factor` is the number of replicas for this instance.

        When using distributed replication this will return the global replication factor.
        """
        return self._pb_ir.getSessionOptions().getGlobalReplicationFactor()

    @replication_factor.setter
    def replication_factor(self, value: int) -> None:
        if self._pb_ir.isPrepared():
            raise RuntimeError(
                f"The `Ir` {self.id} is already prepared, you probably have created a session"
                " associated with this ir and cannot change the replication_factor after this."
            )
        if value > 1:
            self._pb_ir.getSessionOptions().enableReplicatedGraphs = True
        self._pb_ir.getSessionOptions().replicatedGraphCount = value

    @property
    def instance_replication_factor(self) -> int:
        return self._pb_ir.getSessionOptions().replicatedGraphCount

    @property
    def id(self) -> int:
        return self._pb_ir.getId()

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, value: 'Ir') -> bool:
        if not isinstance(value, Ir):
            raise TypeError(
                f"Value must be of type popxl.Ir. Type: {type(value)}. Value: {value}."
            )
        return self.id == value.id

    def __repr__(self) -> str:
        return f"Ir[id={self.id}]"

    def replica_grouping(self,
                         stride: int = 1,
                         group_size: Optional[int] = None
                         ) -> 'ReplicaGrouping':
        """
        Create a :py:class:`~popxl.ReplicaGrouping` object.

        A :py:class:`~popxl.ReplicaGrouping` object represents a way in which
        replicas are grouped for the purpose of getting and setting variable
        values and
        :ref:`collective operations<popxl_ops_collectives_available_ops>`.

        A grouping always exactly partitions a set of replicas, so every replica
        is exactly in one group. We specify these partitions with a ``stride``
        and ``group_size`` argument. The ``stride`` specifies the offset between
        replicas within a group and the ``group_size`` specifies the number of
        replicas within a group.

        Group with ``stride`` 1 and ``group_size`` 2 for 8 replicas):

        .. code-block:: python
            ir.replica_grouping(1, 2).assignment
            [0,0,1,1,2,2,3,3]

        Group with ``stride`` 1 and ``group_size`` 4 for 8 replicas:

        .. code-block:: python
            ir.replica_grouping(1, 4).assignment
            [0,0,0,0,1,1,1,1]

        Group with ``stride`` 2 and ``group_size`` 4 for 8 replicas:

        .. code-block:: python
            ir.replica_grouping(2, 4).assignment
            [0,1,0,1,0,1,0,1]

        Group with ``stride`` 4 and ``group_size`` 2 for 8 replicas:

        .. code-block:: python
            ir.replica_grouping(4, 2).assignment
            [0,1,2,3,0,1,2,3]

        Args:
            stride (int): The offset between elements in a replica group. Defaults to 1.
            group_size (Optional[int]): The number of replicas in each replica group.
                If not provided the `group_size = ir.replication_factor // stride`

        Returns:
            ReplicaGrouping: An object describing the replica grouping.
        """
        return ReplicaGrouping._from_params(ir=self,
                                            stride=stride,
                                            group_size=group_size)
