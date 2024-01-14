import gc
import queue
import sys

import numpy as np
import scipy.special
from typing import Iterable, List
from memory_profiler import profile


class Tensor(object):

    def __init__(self, data,
                 autograd: bool = False,
                 creation_op=None):

        self.data = np.array(data, dtype=np.float64)
        self.shape = self.data.shape
        self.ndim = self.data.ndim
        self.autograd = autograd
        if autograd:
            self.grad = np.zeros_like(self.data)
        else:
            self.grad = None

        self.creation_op = creation_op
        self.dependents = {}

        self.tcg_id = TcGraph.AddTensor(self)

    def all_children_grads_accounted_for(self):
        for cnt in self.dependents.values():
            if cnt != 0:
                return False
        return True

    def backward(self, grad=None, origin_id=None):
        if self.autograd:
            if grad is None:
                self.grad = np.ones_like(self.data)
            else:
                self.grad += grad
                if origin_id is not None:
                    if self.dependents[origin_id] == 0:
                        raise Exception("cannot backprop more than once")
                    self.dependents[origin_id] -= 1

            if self.all_children_grads_accounted_for():
                if self.creation_op is not None:
                    self.creation_op.backward(self.grad)

    def __add__(self, other):
        op: Op = AddOp(self, other)
        return op.calc()

    def __neg__(self):
        op: Op = NegOp(self)
        return op.calc()

    def __sub__(self, other):
        op: Op = SubOp(self, other)
        return op.calc()

    def __mul__(self, other):
        op: Op = MulOp(self, other)
        return op.calc()

    def __truediv__(self, other):
        op: Op = DivOp(self, other)
        return op.calc()

    def __pow__(self, power):
        if not isinstance(power, Tensor):
            power = Tensor(power, autograd=False)
        op: Op = PowOp(self, power)
        return op.calc()

    def mm(self, x):
        op: Op = MatMulOp(self, x)
        return op.calc()

    def bmm(self, x):
        op: Op = BatchMatMul(self, x)
        return op.calc()

    def exp(self):
        op: Op = ExpOp(self)
        return op.calc()

    def log(self):
        op: Op = LogOp(self)
        return op.calc()

    def sin(self):
        op: Op = SinOp(self)
        return op.calc()

    def cos(self):
        op: Op = CosOp(self)
        return op.calc()

    def sigmoid(self):
        op: Op = SigmoidOp(self)
        return op.calc()

    def tanh(self):
        op: Op = TanhOp(self)
        return op.calc()

    def relu(self):
        op: Op = ReLuOp(self)
        return op.calc()

    def softmax(self):
        op: Op = SoftmaxOp(self)
        return op.calc()

    def abs(self):
        op: Op = AbsOp(self)
        return op.calc()

    def sum(self, axes):
        op: Op = SumOp(self, axes)
        return op.calc()

    def max(self, axes):
        op: Op = MaxOp(self, axes)
        return op.calc()

    def mean(self, axes, keepdims=False):
        op: Op = MeanOp(self, axes, keepdims)
        return op.calc()

    def var(self, axes, keepdims=False):
        op: Op = VarOp(self, axes, keepdims)
        return op.calc()

    def sqrt(self):
        op: Op = SqrtOp(self)
        return op.calc()

    def broadcast(self, other):
        if self.shape == other.shape:
            return self, other

        s1 = list(self.shape)
        s2 = list(other.shape)
        if len(s1) > len(s2):
            s2 = [1] * (len(s1) - len(s2)) + s2
            if s1 == s2:
                t = BroadcastOp(other, self.shape).calc()
                """
                @bug_fix_lry
                should return self, t
                return t, other
                """
                return self, t
        else:
            s1 = [1] * (len(s2) - len(s1)) + s1
            if s1 == s2:
                t = BroadcastOp(self, other.shape).calc()
                """
                @bug_fix_lry
                should return self, t
                return self, t
                """
                return t, other

        s = []
        for i in range(len(s1)):
            if s1[i] != s2[i]:
                if s1[i] == 1:
                    s.append(s2[i])
                elif s2[i] == 1:
                    s.append(s1[i])
                else:
                    raise Exception("cannot broadcast")
            else:
                s.append(s1[i])

        if s != list(self.shape):
            t1 = BroadcastOp(self, s).calc()
        else:
            t1 = self

        if s != list(other.shape):
            t2 = BroadcastOp(other, s).calc()
        else:
            t2 = other
        return t1, t2

    def squeeze(self, dim):
        op: Op = SqueezeOp(self, dim)
        return op.calc()

    def unsqueeze(self, dim):
        op: Op = UnsqueezeOp(self, dim)
        return op.calc()

    def transpose(self, axes: Iterable[int] = None):
        op: Op = TransposeOp(self, axes)
        return op.calc()

    def reshape(self, shape):
        op: Op = ReshapeOp(self, shape)
        return op.calc()

    def concat(self, other, axes):
        op: Op = ConcatOp(self, other, axes)
        return op.calc()

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())


class TcGraph:
    instance = None

    def __init__(self):
        self.tmap = dict()
        # (op_name, (input1, input2, ...), (output1, output2, ...))
        self.graph = list()

    @classmethod
    def get_instantce(cls):
        if not cls.instance:
            cls.instance = TcGraph()

        return cls.instance

    @classmethod
    def GetTensor(cls, t):
        return cls.get_instantce().getTensor(t)

    @classmethod
    def Compile(cls):
        return cls.get_instantce().compile()

    @classmethod
    def Clear(cls):
        return cls.get_instantce().clear()

    def compile(self):
        '''
        Convert TcGraph into T-Lang program.
        '''

        graph = self.graph
        tensor_dict = dict(map(reversed, self.tmap.items()))

        op_list = ['def main(){\n']

        tensor_input = set()
        tensor_mid = set()

        for (_, __, out) in graph:
            tensor_mid.update(out)

        for (_, inp, __) in graph:
            tensor_input.update(set(inp).difference(tensor_mid))

        # create all input tensor
        for id in tensor_input:
            t = tensor_dict[id]
            shape = 'x'.join(str(d) for d in t.data.shape)
            data = ', '.join(str(e) for e in t.data.flat)
            op = f'  var v{id}<{shape}> = [{data}];\n'
            op_list.append(op)

        while True:
            # if graph not empty, find all which input all generated.
            is_emitable = False
            for (name, inp, out) in graph:
                assert len(out) == 1 and "for now only support 1 result."
                out = out[0]

                is_emitable = out not in tensor_input and set(inp).issubset(tensor_input)
                params = ', '.join(f'v{tid}' for tid in inp)

                if is_emitable:
                    op = f'  var v{out} = {name}({params});\n'
                    if name in ['add', 'matmul']:
                        assert len(inp) == 2 and 'binop must have 2 op.'
                        # TODO: Add more.
                        binop_dict = {
                            'add': '+',
                            'matmul': '.',
                        }
                        op = f'  var v{out} = v{inp[0]} {binop_dict[name]} v{inp[1]};\n'

                    op_list.append(op)

                    tensor_input.add(out)
                    tensor_mid.remove(out)

                    # if cur op's result is the last, also emit printOp.
                    if len(tensor_mid) == 0:
                        op_list.append(f'  print(v{out});\n')
            if not is_emitable:
                break

        op_list.append('}\n')
        return ''.join(op_list)

    def clear(self):
        self.graph.clear()
        self.tmap.clear()

    def getTensor(self, t):
        '''
        return tensor internal repr id.
        '''
        tmap = self.tmap
        # assert type(t) == Tensor and "getTensor input only suppor Tensor."
        if t not in tmap:
            #print('TcGraph: Warning: current tensor not managed.')
            return self.addTensor(t)

        # assert t.get_tcg_id() == tmap[t] and "tcg_id and id managed in TcGraph must be the same."
        return tmap[t]

    @classmethod
    def AddTensor(cls, t):
        return cls.get_instantce().addTensor(t)

    def addTensor(self, t):
        '''
        alloc a internal repr id for given tensor.
        '''
        tmap = self.tmap
        if t not in tmap:
            tmap[t] = len(tmap)
        return self.getTensor(t)

    @classmethod
    def AddOp(cls, op_name, inputs, outputs):
        return cls.get_instantce().addOp(op_name, inputs, outputs)

    def addOp(self, op_name, inputs, outputs):
        self.graph.append((op_name,
                           tuple(self.getTensor(i) for i in inputs),
                           tuple(self.addTensor(o) for o in outputs)
                           ))


# TODO grad_fn 是否支持静态、动态重载
class Op:

    def __init__(self, args, tcc_opname='unsupported'):
        self.input: List[Tensor] = args
        self.output: [Tensor, None] = None
        self.grad_fn = []

    def calc(self):
        raise NotImplementedError

    def backward(self, grad: np.ndarray):
        assert len(self.input) == len(self.grad_fn)

        for i in range(len(self.input)):
            if self.input[i].autograd:
                self.input[i].backward(self.grad_fn[i](grad, self.output, self.input), id(self.output))

    def add_dependency(self):
        for i in range(len(self.input)):
            output_id = id(self.output)
            if id(self.output) not in self.input[i].dependents:
                self.input[i].dependents[output_id] = 1
            else:
                self.input[i].dependents[output_id] += 1


class AddOp(Op):

    def __init__(self, t1: Tensor, t2: Tensor):
        t1, t2 = t1.broadcast(t2)
        super(AddOp, self).__init__([t1, t2])
        self.grad_fn = [
            lambda grad, out, args: grad * np.ones_like(args[0].data),
            lambda grad, out, args: grad * np.ones_like(args[1].data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self) -> Tensor:
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data + self.input[1].data, creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('add', [self.input[0], self.input[1]], [self.output])
        return self.output


class SubOp(Op):

    def __init__(self, t1: Tensor, t2: Tensor):
        t1, t2 = t1.broadcast(t2)
        super(SubOp, self).__init__([t1, t2])
        self.grad_fn = [
            lambda grad, out, args: grad * np.ones_like(args[0].data),
            lambda grad, out, args: grad * -np.ones_like(args[1].data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self) -> Tensor:
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data - self.input[1].data, creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('sub', [self.input[0], self.input[1]], [self.output])
        return self.output


class MulOp(Op):

    def __init__(self, t1: Tensor, t2: Tensor):
        t1, t2 = t1.broadcast(t2)
        super(MulOp, self).__init__([t1, t2])
        self.grad_fn = [
            lambda grad, out, args: grad * args[1].data,
            lambda grad, out, args: grad * args[0].data
        ]
        self.calc()
        self.add_dependency()

    def calc(self) -> Tensor:
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data * self.input[1].data, creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('mul', [self.input[0], self.input[1]], [self.output])
        return self.output


class DivOp(Op):

    def __init__(self, t1: Tensor, t2: Tensor):
        t1, t2 = t1.broadcast(t2)
        super(DivOp, self).__init__([t1, t2])
        self.grad_fn = [
            lambda grad, out, args: grad / args[1].data,
            lambda grad, out, args: grad * -args[0].data / (args[1].data * args[1].data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self) -> Tensor:
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data / self.input[1].data, creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('div', [self.input[0], self.input[1]], [self.output])
        return self.output


class PowOp(Op):

    def __init__(self, t1: Tensor, t2: Tensor):
        t1, t2 = t1.broadcast(t2)
        super(PowOp, self).__init__([t1, t2])
        self.grad_fn = [
            lambda grad, out, args: grad * args[1].data * np.power(args[0].data, args[1].data - 1),
            lambda grad, out, args: grad * np.log(args[0].data) * np.power(args[0].data, args[1].data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self) -> Tensor:
        if self.output is None:
            self.output: Tensor = Tensor(np.power(self.input[0].data, self.input[1].data), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('pow', [self.input[0], self.input[1]], [self.output])
        return self.output


class NegOp(Op):

    def __init__(self, t1: Tensor):
        super(NegOp, self).__init__([t1])
        self.grad_fn = [
            lambda grad, out, args: grad * -np.ones_like(args[0].data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self) -> Tensor:
        if self.output is None:
            self.output: Tensor = Tensor(-self.input[0].data, creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('neg', [self.input[0]], [self.output])
        return self.output


class MatMulOp(Op):
    """
    the back-propagation of gradient in matrix dot multiplication
    grad(C): the gradient of matrix C
    @: the matrix multiplication
    .T: the transpose of a matrix

    The shapes of matrix A, B, C is (m, n), (n, p), (m, p) respectively
    then
        grad(A) = grad(C) @ B.T
        grad(B) = A.T @ grad(C)

    supports k (k > 2) dimensions of matrix multiplication

    """

    def __init__(self, t1: Tensor, t2: Tensor):
        super(MatMulOp, self).__init__([t1, t2])
        '''
        self.grad_fn = [
            lambda grad, out, args: grad @ args[1].data.transpose(),
            lambda grad, out, args: args[0].data.transpose() @ grad
        ]
        '''
        self.grad_fn = [
            lambda grad, out, args: (grad @ args[1].data.transpose(list(range(args[1].data.ndim - 2)) + [-1, -2]))
            .sum(tuple(range(grad.ndim - args[0].data.ndim))),
            lambda grad, out, args: (args[0].data.transpose(list(range(args[0].data.ndim - 2)) + [-1, -2]) @ grad)
            .sum(tuple(range(grad.ndim - args[1].data.ndim)))
        ]
        self.calc()
        self.add_dependency()

    def calc(self) -> Tensor:
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data.dot(self.input[1].data), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('matmul', [self.input[0], self.input[1]], [self.output])
        return self.output


class BatchMatMul(Op):
    """
    batch * batch
    [B, L1, m] * [B, m, L2]
    first split to [L1, m] and [m, L2]
    then list ([L1, m].mm([m, L2]) -> [L1, L2]), len(list) = B
    last concatenate list -> [B, L1, L2]
    """

    def __init__(self, t1: Tensor, t2: Tensor):
        super(BatchMatMul, self).__init__([t1, t2])
        assert len(self.input[0].shape) == len(self.input[1].shape)
        for i in range(len(self.input[0].shape) - 2):
            assert self.input[0].shape[i] == self.input[1].shape[i]
        assert self.input[0].shape[-1] == self.input[1].shape[-2]
        self.B = self.input[0].shape[0]
        self.L1 = self.input[0].shape[1]
        self.m = self.input[0].shape[2]
        self.L2 = self.input[1].shape[2]
        self.grad_fn = [
            lambda grad, out, args: np.concatenate(
                [(_grad.reshape(self.L1, self.L2) @ _arg1.reshape(self.m, self.L2).transpose())
                 .reshape(1, self.L1, self.m) for (_arg1, _grad)
                 in zip(np.split(args[1].data, len(args[1].data), axis=0), np.split(grad, len(grad), axis=0))], axis=0),
            lambda grad, out, args: np.concatenate(
                [(_arg0.reshape(self.L1, self.m).transpose() @ _grad.reshape(self.L1, self.L2))
                 .reshape(1, self.m, self.L2) for (_arg0, _grad)
                 in zip(np.split(args[0].data, len(args[0].data), axis=0), np.split(grad, len(grad), axis=0))], axis=0),
        ]
        self.calc()
        self.add_dependency()

    def calc(self) -> Tensor:
        if self.output is None:
            mats1 = np.split(self.input[0].data, len(self.input[0].data), axis=0)
            mats2 = np.split(self.input[1].data, len(self.input[1].data), axis=0)
            rs = [(m1.reshape(self.L1, -1).dot(m2.reshape(-1, self.L2)).reshape(1, self.L1, self.L2)) for (m1, m2) in zip(mats1, mats2)]
            self.output: Tensor = Tensor(np.concatenate(rs, axis=0), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
        return self.output


class ExpOp(Op):
    def __init__(self, t: Tensor):
        super(ExpOp, self).__init__([t])
        self.grad_fn = [
            lambda grad, out, args: grad * out.data
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(np.exp(self.input[0].data), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('exp', [self.input[0]], [self.output])
        return self.output


class LogOp(Op):
    def __init__(self, t: Tensor):
        super(LogOp, self).__init__([t])
        self.grad_fn = [
            lambda grad, out, args: grad / args[0].data
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(np.log(self.input[0].data), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('log', [self.input[0]], [self.output])
        return self.output


class SinOp(Op):
    def __init__(self, t: Tensor):
        super(SinOp, self).__init__([t])
        self.grad_fn = [
            lambda grad, out, args: grad * np.cos(args[0].data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(np.sin(self.input[0].data), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('sin', [self.input[0]], [self.output])
        return self.output


class CosOp(Op):
    def __init__(self, t: Tensor):
        super(CosOp, self).__init__([t])
        self.grad_fn = [
            lambda grad, out, args: grad * -np.sin(args[0].data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(np.cos(self.input[0].data), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('cos', [self.input[0]], [self.output])
        return self.output


class SigmoidOp(Op):

    def __init__(self, t: Tensor):
        super(SigmoidOp, self).__init__([t])
        self.grad_fn = [
            lambda grad, out, args: grad * out.data * (1 - out.data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(1 / (1 + np.exp(-self.input[0].data)), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('sigmoid', [self.input[0]], [self.output])
        return self.output


class TanhOp(Op):

    def __init__(self, t: Tensor):
        super(TanhOp, self).__init__([t])
        self.grad_fn = [
            lambda grad, out, args: grad * (1 - out.data * out.data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(np.tanh(self.input[0].data), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('tanh', [self.input[0]], [self.output])
        return self.output


class ReLuOp(Op):
    def __init__(self, t: Tensor):
        super(ReLuOp, self).__init__([t])
        self.grad_fn = [
            lambda grad, out, args: grad * (args[0].data > 0)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(np.maximum(self.input[0].data, 0), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('relu', [self.input[0]], [self.output])
        return self.output


class AbsOp(Op):
    def __init__(self, t: Tensor):
        super(AbsOp, self).__init__([t])
        self.grad_fn = [
            lambda grad, out, args: grad * np.sign(args[0].data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(np.abs(self.input[0].data), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('abs', [self.input[0]], [self.output])
        return self.output


def reduce_shape(shape: tuple, axes: [None, int, Iterable]):
    """
    :param shape: (3, 4, 5, 6) for example
    :param axes: (0, 2, 3) for example
    :return: axes = (0, 2, 3), _shape = (1, 4, 1, 1) for example
    """
    if axes is None:
        return None, (1,) * len(shape)

    _shape = list(shape)
    if isinstance(axes, int):
        axes = [axes]
    else:
        axes = list(axes)

    for i in range(len(axes)):
        if axes[i] < 0:
            axes[i] += len(shape)
        _shape[axes[i]] = 1

    axes = tuple(axes)

    _shape = tuple(_shape)
    return axes, _shape


def concat_shape(shapes: List[tuple], axes: [None, int]):
    if axes is None:
        return None, shapes[0]

    _shapes = [list(x) for x in shapes]

    concat_sum = 0
    for _shape in _shapes:
        concat_sum += _shape[axes]

    _shapes[0][axes] = concat_sum

    return axes, tuple(_shapes[0])


class SumOp(Op):

    def __init__(self, t: Tensor, axes: [int, Iterable]):
        super(SumOp, self).__init__([t])

        self.axes, self._shape = reduce_shape(t.shape, axes)

        self.grad_fn = [
            # there will be broadcast
            lambda grad, out, args: grad.reshape(self._shape) * np.ones_like(args[0].data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data.sum(axis=self.axes), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('sum', [self.input[0]], [self.output])
        return self.output


class MaxOp(Op):

    def __init__(self, t: Tensor, axes: [int, Iterable]):
        super(MaxOp, self).__init__([t])
        """
        @bug_fix_lry
        args[0].data == out.data
        
            seems to be not appropriate
            changes to be
            
        self.axes, self._shape = reduce_shape(t.shape, axes)
        self.grad_fn = [
            lambda grad, out, args: grad.reshape(self._shape) * (args[0].data == out.data.reshape(self._shape))
        ]
        """
        self.axes, self._shape = reduce_shape(t.shape, axes)
        self.grad_fn = [
            lambda grad, out, args: grad.reshape(self._shape) * (args[0].data == out.data.reshape(self._shape))
        ]
        self.axes = axes
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data.max(axis=self.axes), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('max', [self.input[0]], [self.output])
        return self.output


class MeanOp(Op):

    def __init__(self, t: Tensor, axes: [int, Iterable], keepdims):
        super(MeanOp, self).__init__([t])

        self.keepdims = keepdims
        self.axes, self._shape = reduce_shape(t.shape, axes)

        self.N = 1
        for axis in self.axes:
            self.N *= t.shape[axis]

        self.grad_fn = [
            lambda grad, out, args: grad.reshape(self._shape) * np.ones_like(args[0].data) / self.N
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data.sum(axis=self.axes, keepdims=self.keepdims) / self.N,
                                         creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('mean', [self.input[0]], [self.output])
        return self.output


class VarOp(Op):
    """
    y_i = 1/n * [(x_i1 - x_i_mean)^2 + (x_i2 - x_i_mean)^2 + ... + (x_in - x_i_mean)^2]
    dy_i / dx_ij = 2 * 1/n * (x_ij - x_i_mean)
    """

    def __init__(self, t: Tensor, axes: [int, Iterable], keepdims: bool):
        super(VarOp, self).__init__([t])

        self.keepdims = keepdims
        self.axes, self._shape = reduce_shape(t.shape, axes)

        self.N = 1
        for axis in self.axes:
            self.N *= t.shape[axis]

        self.grad_fn = [
            lambda grad, out, args: grad.reshape(self._shape) *
                                    2 * (args[0].data - args[0].data.sum(self.axes, keepdims=True) / self.N) / self.N
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            data = self.input[0].data
            mean_val = data.sum(axis=self.axes, keepdims=True) / self.N
            data = data - mean_val
            data = data * data
            self.output: Tensor = Tensor(data.sum(axis=self.axes, keepdims=self.keepdims) / self.N, creation_op=self,
                                         autograd=any(t.autograd for t in self.input))

        return self.output


class SqrtOp(Op):
    def __init__(self, t: Tensor):
        super(SqrtOp, self).__init__([t])
        self.grad_fn = [
            lambda grad, out, args: grad * 0.5 * (1 / out.data)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(np.sqrt(self.input[0].data), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
        return self.output


# TODO softmax 可以指定维度
class SoftmaxOp(Op):

    def __init__(self, t: Tensor):
        super(SoftmaxOp, self).__init__([t])
        self.grad_fn = [
            lambda grad, out, args: out.data * (grad - np.sum(grad * out.data, axis=-1, keepdims=True))
        ]
        self.calc()
        self.add_dependency()

    # TODO: TcGraph: argument config is needed.
    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(scipy.special.softmax(self.input[0].data, axis=-1), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('softmax', [self.input[0]], [self.output])
        return self.output


class BroadcastOp(Op):
    def __init__(self, t: Tensor, shape: [int]):
        super(BroadcastOp, self).__init__([t])
        self.shape = shape
        self.axes = []
        if len(shape) > len(t.shape):
            self.axes = list(range(len(shape) - len(t.shape)))

        offset = len(shape) - len(t.shape)
        for i in range(len(t.shape)):
            if t.shape[i] != shape[i + offset]:
                self.axes.append(i + offset)

        self.axes = tuple(self.axes)
        self.grad_fn = [
            lambda grad, out, args: grad.sum(axis=self.axes).reshape(args[0].shape)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(np.broadcast_to(self.input[0].data, self.shape), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('broadcast', [self.input[0]], [self.output])
        return self.output


class SqueezeOp(Op):
    """
    squeeze will delete the dimension of value 1
    axis must be the dimension of value 1
    """

    def __init__(self, t: Tensor, axis: int):
        super(SqueezeOp, self).__init__([t])
        self.axis = axis
        self.grad_fn = [
            lambda grad, out, args: grad.reshape(args[0].data.shape)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(np.squeeze(self.input[0].data, axis=self.axis), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('squeeze', [self.input[0]], [self.output])
        return self.output


class UnsqueezeOp(Op):
    """
    insert a dimension of value 1 at the position of axis
    """

    def __init__(self, t: Tensor, axis: int):
        super(UnsqueezeOp, self).__init__([t])
        self.axis = axis
        self.grad_fn = [
            lambda grad, out, args: grad.reshape(args[0].data.shape)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(np.expand_dims(self.input[0].data, axis=self.axis), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('unsqueeze', [self.input[0]], [self.output])
        return self.output


class TransposeOp(Op):

    def __init__(self, t: Tensor, axes: Iterable[int] = None):
        super(TransposeOp, self).__init__([t])
        if axes is None:
            self.axes = list(range(len(t.shape) - 1, -1, -1))
        else:
            self.axes = axes
        self.grad_fn = [
            lambda grad, out, args: grad.transpose(
                sorted(range(len(self.axes)), key=lambda x: self.axes[x])
            )
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data.transpose(self.axes), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('transpose', [self.input[0]], [self.output])
        return self.output


class ReshapeOp(Op):
    def __init__(self, t: Tensor, shape: [int]):
        super(ReshapeOp, self).__init__([t])

        shape = list(shape)
        for i in range(len(shape)):
            if shape[i] == -1:
                shape[i] = int(np.prod(t.shape) / np.prod(shape))
                break
        shape = tuple(shape)

        self.shape = shape
        self.grad_fn = [
            lambda grad, out, args: grad.reshape(args[0].data.shape)
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(self.input[0].data.reshape(self.shape), creation_op=self,
                                         autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('reshape', [self.input[0]], [self.output])
        return self.output


class ConcatOp(Op):
    def __init__(self, t_list: List[Tensor], axis: [None, int]):
        super(ConcatOp, self).__init__(t_list)
        shapes = [x.shape for x in t_list]
        self.axis, self._shape = concat_shape(shapes, axis)

        self.grad_fn = [
            lambda grad, out, args: np.split(grad, len(t_list), axis=axis)[i] for i in range(len(t_list))
        ]
        self.calc()
        self.add_dependency()

    def calc(self):
        if self.output is None:
            concat_data = self.input[0].data
            for index, t in enumerate(self.input):
                if index != len(self.input) - 1:
                    concat_data = np.concatenate((concat_data, self.input[index + 1].data), axis=self.axis)
            self.output: Tensor = Tensor(concat_data, creation_op=self, autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('concat', self.input, [self.output])
        return self.output


class Conv2dOp(Op):
    def __init__(self, t: Tensor, kernel: Tensor, stride: int, padding: int):
        super(Conv2dOp, self).__init__([t, kernel])
        self.stride = stride
        self.padding = padding
        self.grad_fn = [
            lambda grad, out, args: self.calc_grad_input(grad),
            lambda grad, out, args: self.calc_grad_kernel(grad)
        ]
        self.calc()
        self.add_dependency()

    # @profile()
    def calc(self):
        if self.output is None:
            self.img_cols, data = self._conv2d(self.input[0].data, self.input[1].data, self.stride, self.padding)
            self.output: Tensor = Tensor(data, creation_op=self, autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('conv2d', [self.input[0], self.input[1]], [self.output])
        return self.output

    # @profile()
    def calc_grad_kernel(self, grad_output: np.ndarray):
        B, _, _, _ = grad_output.shape  # grad: (B, O, out_w, out_h)
        # kernel 的 grad
        YC_grad = self._y_to_YC(grad_output)  # YC_grad:(B * out_w * out_h, O)
        XC_T = self.img_cols.transpose()  # img_cols: (B * out_w * out_h, C * K * K)
        # X_CT: (C * K * K, B * out_w * out_h)
        weight_grad = np.dot(XC_T, YC_grad)  # weight_grad: (C(I) * K * K, O)
        return weight_grad.transpose().reshape(self.input[1].data.shape)  # out: (O, C(I), K, K)

    def calc_grad_input(self, grad_output: np.ndarray):
        # input 的 grad
        grad_output_padding = self._stride_and_padding(grad_output,
                                                       stride=self.stride,
                                                       padding=self.input[1].data.shape[-1] - 1)
        kernel_ = self._rotate180(self.input[1].data)
        grad_input = self._conv2d(grad_output_padding, kernel_, stride=1)[1]
        if self.padding != 0:
            grad_input = grad_input[:, :, self.padding:-self.padding, self.padding:-self.padding]
        return grad_input

    # @profile()
    def _conv2d(self, img: np.ndarray, kernel: np.ndarray, stride=1, padding=0):
        """
        out_H = (H - K) // stride + 1
        out_W = (W - K) // stride + 1
        :param img: (B, C, H, W)
        :param kernel: (O, I, K, K)
        :param stride:
        :param padding:
        :return: img_cols: (B * out_H * out_W, O)  result: (B, O, out_H, out_W)
        """
        if padding != 0:
            img = np.pad(img, tuple((
                (0, 0),
                (0, 0),
                (padding, padding),
                (padding, padding))),
                         'constant',
                         constant_values=0)
        B, _, H, W = img.shape
        O, I, K, K = kernel.shape
        img_cols = self._img2col(img, kernel_size=K, stride=stride)
        # 将 (B, C, H, W) 维的img扩展为(B * out_h * out_h, C * K * K) 维的行向量组
        kernel_ = kernel.reshape(O, -1).T  # 将(O, I, K, K)维的kernel展平为 (C * K * K, O)维的列向量组
        y = np.dot(img_cols, kernel_)  # 矩阵相乘 y的维度为 (B * out_h * out_h, O)
        output_H, output_W = (H - K) // stride + 1, (W - K) // stride + 1
        result = self._YC_to_y(y, B, O, output_H, output_W)  # reshape变换输出的形式
        return img_cols, result

    # YC变换成Y，前向过程需要
    # YC的维度为 (B * out_h * out_h) * O
    # y的维度为
    def _YC_to_y(self, YC, batch_size, channel, output_H, output_W):
        result = YC.reshape((batch_size, YC.shape[0] // batch_size, -1)).reshape(
            (batch_size, output_H, output_W, channel))
        return result.transpose((0, 3, 1, 2))

    # y变换成YC，反向传播过程需要
    def _y_to_YC(self, y):
        B, C, H, W = y.shape
        result = y.transpose((0, 2, 3, 1)).reshape(B * W * H, -1)
        return result

    def _img2col(self, img: np.ndarray, kernel_size, stride=1):
        B, C, H, W = img.shape
        out_h = (H - kernel_size) // stride + 1
        out_w = (W - kernel_size) // stride + 1

        col = np.zeros((B, C, kernel_size, kernel_size, out_h, out_w))

        for y in range(kernel_size):
            y_max = y + stride * out_h
            for x in range(kernel_size):
                x_max = x + stride * out_w
                col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

        col = np.ascontiguousarray(col.transpose((0, 4, 5, 1, 2, 3))).reshape(B * out_h * out_w, -1)
        # (B * out_h * out_w, C * kernel_size * kernel_size)
        return col

    def _stride_and_padding(self, grad_output, stride, padding):
        """
        对于stride=1，直接在梯度矩阵周围(stride - 1)层填充0
        对于stride>1，先在梯度矩阵中间每行和每列之间分别插入(stride - 1)层的0，然后再在梯度矩阵周围(stride - 1)层填充0
        :param grad_output: shape = (B, O, out_h, out_w)
        :param stride:
        :param padding:
        :return:
        """
        if stride > 1:
            N, O, output_H, output_W = grad_output.shape
            inserted_H, inserted_W = output_H + (output_H - 1) * (stride - 1), output_W + (output_W - 1) * (stride - 1)
            inserted_eta = np.zeros((N, O, inserted_H, inserted_W))
            inserted_eta[:, :, ::stride, ::stride] = grad_output
            grad_output = inserted_eta

        grad_output = np.lib.pad(grad_output, ((0, 0),
                                               (0, 0),
                                               (padding, padding),
                                               (padding, padding)), "constant",
                                 constant_values=0)
        return grad_output

    def _rotate180(self, kernel):
        # 旋转90+90度构成旋转180度
        # weight = np.rot90(weight, axes=(2, 3))
        # weight = np.rot90(weight, axes=(2, 3))
        _, C, _, _ = kernel.shape
        weight_flip = np.flip(kernel, (2, 3))  # 卷积核旋转180度
        weight_flip_swap = np.swapaxes(weight_flip, 0, 1)  # 交换输入、输出通道的顺序[C,O,H,W]
        return weight_flip_swap


class MaxPool2dOp(Op):
    def __init__(self, t: Tensor, kernel_size: int, stride: int, padding: int):
        super(MaxPool2dOp, self).__init__([t])
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.grad_fn = [
            lambda grad, out, args: self.max_pool2d_grad_input(grad, args[0].data, out.data,
                                                               self.kernel_size, self.stride,
                                                               self.padding)
        ]
        self.calc()
        self.add_dependency()

    def max_pool2d_grad_input(self, grad: np.ndarray, input: np.ndarray, output: np.ndarray, kernel_size: int,
                              stride: int, padding: int):
        batch_size, in_channels, in_height, in_width = input.shape
        out_height = (in_height - kernel_size + 2 * padding) // stride + 1
        out_width = (in_width - kernel_size + 2 * padding) // stride + 1
        grad_input = np.zeros(input.shape)
        for b in range(batch_size):
            for c in range(in_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        grad_input[b, c, h * stride:h * stride + kernel_size, w * stride:w * stride + kernel_size] += \
                            grad[b, c, h, w] * (
                                    input[b, c, h * stride:h * stride + kernel_size,
                                    w * stride:w * stride + kernel_size] ==
                                    output[b, c, h, w])
        return grad_input

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(
                self.max_pool2d(self.input[0].data, self.kernel_size, self.stride, self.padding), creation_op=self,
                autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('max_pool2d', [self.input[0]], [self.output])
        return self.output

    def max_pool2d(self, input: np.ndarray, kernel_size: int, stride: int, padding: int):
        batch_size, in_channels, in_height, in_width = input.shape
        out_height = (in_height - kernel_size + 2 * padding) // stride + 1
        out_width = (in_width - kernel_size + 2 * padding) // stride + 1
        output = np.zeros((batch_size, in_channels, out_height, out_width))
        for b in range(batch_size):
            for c in range(in_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        output[b, c, h, w] = np.max(
                            input[b, c, h * stride:h * stride + kernel_size, w * stride:w * stride + kernel_size])
        return output


class AvgPool2dOp(Op):
    def __init__(self, t: Tensor, kernel_size: int, stride: int, padding: int):
        super(AvgPool2dOp, self).__init__([t])
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.grad_fn = [
            lambda grad, out, args: self.avg_pool2d_grad_input(grad, args[0].data, out.data,
                                                               self.kernel_size, self.stride,
                                                               self.padding)
        ]
        self.calc()
        self.add_dependency()

    def avg_pool2d_grad_input(self, grad: np.ndarray, input: np.ndarray, output: np.ndarray, kernel_size: int,
                              stride: int, padding: int):
        batch_size, in_channels, in_height, in_width = input.shape
        out_height = (in_height - kernel_size + 2 * padding) // stride + 1
        out_width = (in_width - kernel_size + 2 * padding) // stride + 1
        grad_input = np.zeros(input.shape)
        for b in range(batch_size):
            for c in range(in_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        grad_input[b, c, h * stride:h * stride + kernel_size, w * stride:w * stride + kernel_size] += \
                            grad[b, c, h, w]
        return grad_input / (kernel_size * kernel_size)

    def calc(self):
        if self.output is None:
            self.output: Tensor = Tensor(
                self.avg_pool2d(self.input[0].data, self.kernel_size, self.stride, self.padding), creation_op=self,
                autograd=any(t.autograd for t in self.input))
            TcGraph.AddOp('avg_pool2d', [self.input[0]], [self.output])
        return self.output

    def avg_pool2d(self, input: np.ndarray, kernel_size: int, stride: int, padding: int):
        batch_size, in_channels, in_height, in_width = input.shape
        out_height = (in_height - kernel_size + 2 * padding) // stride + 1
        out_width = (in_width - kernel_size + 2 * padding) // stride + 1
        output = np.zeros((batch_size, in_channels, out_height, out_width))
        for b in range(batch_size):
            for c in range(in_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        output[b, c, h, w] = np.mean(
                            input[b, c, h * stride:h * stride + kernel_size, w * stride:w * stride + kernel_size])
        return output


def log(t: Tensor) -> Tensor:
    return t.log()


def exp(t: Tensor) -> Tensor:
    return t.exp()


def sin(t: Tensor) -> Tensor:
    return t.sin()


def cos(t: Tensor) -> Tensor:
    return t.cos()


def tanh(t: Tensor) -> Tensor:
    return t.tanh()


def sigmoid(t: Tensor) -> Tensor:
    return t.sigmoid()


def relu(t: Tensor) -> Tensor:
    return t.relu()


def mm(t1: Tensor, t2: Tensor) -> Tensor:
    return t1.mm(t2)


def softmax(t: Tensor) -> Tensor:
    return t.softmax()


def abs(t: Tensor) -> Tensor:
    return t.abs()


def sum(t: Tensor, axes: [int, Iterable]) -> Tensor:
    return t.sum(axes)


def max(t: Tensor, dim: int) -> Tensor:
    return t.max(dim)


def mean(t: Tensor, axes: [int, Iterable]) -> Tensor:
    return t.mean(axes)


def var(t: Tensor, axes: [int, Iterable]) -> Tensor:
    return t.var(axes)


def sqrt(t: Tensor) -> Tensor:
    return t.sqrt()


def concat(t_list: List[Tensor], axis: [None, int]) -> Tensor:
    return ConcatOp(t_list, axis).calc()


def conv2d(t: Tensor, kernel: Tensor, stride: int, padding: int) -> Tensor:
    return Conv2dOp(t, kernel, stride, padding).calc()


def max_pool2d(t: Tensor, kernel_size: int, stride: int, padding: int) -> Tensor:
    return MaxPool2dOp(t, kernel_size, stride, padding).calc()


def avg_pool2d(t: Tensor, kernel_size: int, stride: int, padding: int) -> Tensor:
    return AvgPool2dOp(t, kernel_size, stride, padding).calc()


def clear_non_parameter_tensor(root: Tensor):
    tensor_queue: queue.Queue[Tensor] = queue.Queue()
    tensor_queue.put(root)
    while not tensor_queue.empty():
        t = tensor_queue.get()
        if t.creation_op is not None:
            for input_tensor in t.creation_op.input:
                tensor_queue.put(input_tensor)
            t.creation_op.input.clear()
            t.creation_op.output = None
            t.creation_op = None
