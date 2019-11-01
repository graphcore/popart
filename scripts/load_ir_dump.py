import json
import sys


class Consumers:
    def __init__(self):
        self._ops = []

    def _append(self, op):
        self._ops.append(op)

    def __iter__(self):
        yield from iter(self._ops)

    def pipeline_stages(self):
        stages = set()
        for op in self._ops:
            stages.add(op.get_pipeline_stage())
        return stages


class SearchHelper:
    def __init__(self):
        self.seen = set()
        self.front = list()

    def add(self, value):
        if value not in self.seen:
            self.seen.add(value)
            self.front.append(value)

    def __iter__(self):
        while self.front:
            yield self.front.pop()


class Tensor:
    def __init__(self, d):
        self.name = d['name']
        self.consumers = Consumers()
        self.producer = None

    def __str__(self):
        return f'Tensor({self.name})'

    def __repr__(self):
        return str(self)

    def set_producer(self, op):
        if self.producer is not None:
            raise SystemError(f'{str(self)} already has a producer')
        self.producer = op

    def trace_children(self, generations=1):
        front = [self]
        result = {}
        for current_generation in range(generations):
            children = []
            for tensor in front:
                for consumer in tensor.consumers:
                    for output in consumer.outputs.values():
                        children.append(output)
            front = children
            result[current_generation + 1] = children
        return result

    def pipeline_stages(self):
        stages = self.consumers.pipeline_stages()
        if self.producer:
            if self.producer.type == 'IpuCopy':
                stages.add(self.producer.get_pipeline_stage() + 1)
            else:
                stages.add(self.producer.get_pipeline_stage())
        return stages

    def walk_consumers(self, func):
        front = SearchHelper()

        def add_consumers(tensor):
            for consumer in tensor.consumers:
                if func(consumer):
                    front.add(consumer)

        add_consumers(self)
        for op in front:
            for output in op.outputs.values():
                add_consumers(output)

    def walk_and_collect_consumers(self, condition):
        result = set()
        front = SearchHelper()

        def add_consumers(tensor):
            for consumer in tensor.consumers:
                if condition(consumer):
                    front.add(consumer)
                    result.add(consumer)

        add_consumers(self)
        for op in front:
            for output in op.outputs.values():
                add_consumers(output)

        return list(result)


class Op:
    def __init__(self, d, graph):
        self._dict = d
        self.type = d['type']
        self.name = d['name']
        self.attributes = d['attributes']

        self.inputs = {}
        for i in d['inputs']:
            idx = int(i['index'])
            tensor = graph.get_tensor(i)
            tensor.consumers._append(self)
            self.inputs[idx] = tensor

        self.outputs = {}
        for i in d['outputs']:
            idx = int(i['index'])
            tensor = graph.get_tensor(i)
            tensor.set_producer(self)
            self.outputs[idx] = tensor

    def _format_tensor_dict(self, tensor_dict):
        tensor_dict = {k: v.name for k, v in tensor_dict.items()}
        return str(tensor_dict)

    def __str__(self):
        header = f'{self.type}:'
        inputs = f'  Inputs: {self._format_tensor_dict(self.inputs)}'
        outputs = f'  Outputs: {self._format_tensor_dict(self.outputs)}'
        attributes = [f'    {k}: {v}' for k, v in self.attributes.items()]
        attributes = '  Attributes:\n' + '\n'.join(attributes)
        return f'{header}\n{inputs}\n{outputs}\n{attributes}'

    def __repr__(self):
        source = f'{self._format_tensor_dict(self.inputs)}'
        dest = f'{self._format_tensor_dict(self.outputs)}'
        return f'Op({self.type}, {source} -> {dest})'

    def get_pipeline_stage(self):
        return int(self.attributes['__pipeline_stage'])


class Graph:
    def __init__(self, name, ops):
        self.name = name
        self.tensors = {}
        self.ops = [Op(op, self) for op in ops]

        def format_tensor_name(name):
            name = name.replace(':', '_')
            name = name.replace('/', '_')
            return name

        # Add members .tensor_<TensorName> to the graph
        # Allow use of tab auto completion to get a tensor from the ir
        for t in self.tensors.keys():
            name = format_tensor_name(t)
            setattr(self, f'tensor_{name}', self.tensors[t])

    def get_tensor(self, d):
        name = d['name']
        if name not in self.tensors:
            t = Tensor(d)
            self.tensors[name] = t

        return self.tensors[name]

    def __str__(self):
        return (f'Graph(name: {self.name}, ops: {len(self.ops)}'
                ', tensors: {len(self.tensors)})')

    def __repr__(self):
        x = f'Graph({self.name}):'
        x += '\n  ops:'
        first = True
        for op in self.ops:
            lines = str(op).splitlines()
            lines = [f'    {i}' for i in lines]
            lines = '\n'.join(lines)
            if first:
                x += f'{lines}'
                first = False
            else:
                x += f'\n\n{lines}'

        return x


class Ir:
    def __init__(self, ir):
        self.graphs = {}
        for name, graph in ir.items():
            self.graphs[name] = Graph(name, graph)

        # Add members .graph_<GraphName> to the ir
        # Allow use of tab auto completion to get a graph from the ir
        for name in ir.keys():
            setattr(self, f'graph_{name}', self.graphs[name])

    def __str__(self):
        return f'Ir(graphs: {len(self.graphs)})'

    def __repr__(self):
        x = f'Ir:'
        for graph in self.graphs.keys():
            x += f'\n  {graph}'
        return x

    def main_graph(self):
        return self.graphs['maingraph']


def load_from_string(x):
    ir = json.loads(x)
    return Ir(ir)


def load_from_file(x):
    with open(x, 'r') as f:
        y = f.read()
        ir = load_from_string(y)
    return ir


def print_usage():
    print(f'Usage:')
    print(f'  python3 -i {sys.argv[0]} path_to_ir_dump.json')
    print()
    print(f"This will start an interactive python session, with the "
          "ir dump loaded to the variable `ir'")


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print_usage()
    else:
        ir = load_from_file(sys.argv[1])
        print(f'ir = load_from_file({sys.argv[1]})')
