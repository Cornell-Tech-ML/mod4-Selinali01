"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend

from typing import Any

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the negation of a tensor.

        Args:
        ----
        ctx : Context
            The context for saving tensors for the backward pass.
        t1 : Tensor
            The input tensor.

        Returns:
        -------
        Tensor
            The negation of the input tensor.

        """
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient for the negation operation.

        Args:
        ----
        ctx : Context
            The context containing saved tensors from the forward pass.
        grad_output : Tensor
            The gradient of the output tensor.

        Returns:
        -------
        Tensor
            The gradient of the input tensor.

        """
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        """Compute the inverse of a tensor.

        Args:
        ----
        ctx : Context
            The context for saving tensors for the backward pass.
        t1 : Tensor
            The input tensor.

        Returns:
        -------
        Tensor
            The inverse of the input tensor.

        """
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient for the inverse operation.

        Args:
        ----
        ctx : Context
            The context containing saved tensors from the forward pass.
        grad_output : Tensor
            The gradient of the output tensor.

        Returns:
        -------
        Tensor
            The gradient of the input tensor.

        """
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Compute the element-wise addition of two tensors.

        Args:
        ----
        ctx : Context
            The context for saving tensors for the backward pass.
        t1 : Tensor
            The first input tensor.
        t2 : Tensor
            The second input tensor.

        Returns:
        -------
        Tensor
            The result of element-wise addition of t1 and t2.

        """
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradient for the addition operation.

        Args:
        ----
        ctx : Context
            The context containing saved tensors from the forward pass.
        grad_output : Tensor
            The gradient of the output tensor.

        Returns:
        -------
        Tuple[Tensor, Tensor]
            The gradients of the input tensors.

        """
        return grad_output, grad_output


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Return 1 if all are true"""
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)


# TODO: Implement for Task 2.3.
##### START
class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Compute the element-wise multiplication of two tensors.

        Args:
        ----
        ctx : Context
            The context for saving tensors for the backward pass.
        a : Tensor
            The first input tensor.
        b : Tensor
            The second input tensor.

        Returns:
        -------
        Tensor
            The result of element-wise multiplication of a and b.

        """
        ctx.save_for_backward(a, b)
        return a.f.mul_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradient for the multiplication operation.

        Args:
        ----
        ctx : Context
            The context containing saved tensors from the forward pass.
        grad_output : Tensor
            The gradient of the output tensor.

        Returns:
        -------
        Tuple[Tensor, Tensor]
            The gradients of the input tensors.

        """
        (a, b) = ctx.saved_values
        return a.f.mul_zip(b, grad_output), b.f.mul_zip(a, grad_output)


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Compute the sigmoid function element-wise on a tensor.

        Args:
        ----
        ctx : Context
            The context for saving tensors for the backward pass.
        a : Tensor
            The input tensor.

        Returns:
        -------
        Tensor
            The result of applying the sigmoid function to the input tensor.

        """
        out = a.f.sigmoid_map(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient for the sigmoid operation.

        Args:
        ----
        ctx : Context
            The context containing saved tensors from the forward pass.
        grad_output : Tensor
            The gradient of the output tensor.

        Returns:
        -------
        Tensor
            The gradient of the input tensor.

        """
        (t1,) = ctx.saved_values

        return grad_output.f.mul_zip(
            grad_output,
            grad_output.f.mul_zip(
                grad_output.f.add_zip(
                    minitorch.Tensor.make([1.0], (1,), backend=grad_output.backend),
                    grad_output.f.neg_map(t1),
                ),
                t1,
            ),
        )


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Compute the ReLU function element-wise on a tensor.

        Args:
        ----
        ctx : Context
            The context for saving tensors for the backward pass.
        a : Tensor
            The input tensor.

        Returns:
        -------
        Tensor
            The result of applying the ReLU function to the input tensor.

        """
        ctx.save_for_backward(a)
        return a.f.relu_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient for the ReLU operation.

        Args:
        ----
        ctx : Context
            The context containing saved tensors from the forward pass.
        grad_output : Tensor
            The gradient of the output tensor.

        Returns:
        -------
        Tensor
            The gradient of the input tensor.

        """
        (a,) = ctx.saved_values
        return grad_output.f.relu_back_zip(a, grad_output)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Compute the natural logarithm element-wise on a tensor.

        Args:
        ----
        ctx : Context
            The context for saving tensors for the backward pass.
        a : Tensor
            The input tensor.

        Returns:
        -------
        Tensor
            The result of applying the natural logarithm to the input tensor.

        """
        ctx.save_for_backward(a)
        return a.f.log_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient for the logarithm operation.

        Args:
        ----
        ctx : Context
            The context containing saved tensors from the forward pass.
        grad_output : Tensor
            The gradient of the output tensor.

        Returns:
        -------
        Tensor
            The gradient of the input tensor.

        """
        (a,) = ctx.saved_values
        return grad_output.f.log_back_zip(a, grad_output)


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Compute the exponential function element-wise on a tensor.

        Args:
        ----
        ctx : Context
            The context for saving tensors for the backward pass.
        a : Tensor
            The input tensor.

        Returns:
        -------
        Tensor
            The result of applying the exponential function to the input tensor.

        """
        out = a.f.exp_map(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Compute the gradient for the exponential operation.

        Args:
        ----
        ctx : Context
            The context containing saved tensors from the forward pass.
        grad_output : Tensor
            The gradient of the output tensor.

        Returns:
        -------
        Tensor
            The gradient of the input tensor.

        """
        (out,) = ctx.saved_values
        return grad_output * out


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        """Compute the sum of all elements in a tensor along a specified dimension.

        Args:
        ----
        ctx : Context
            The context for saving tensors for the backward pass.
        a : Tensor
            The input tensor.
        dim : Tensor
            The dimension along which to compute the sum.

        Returns:
        -------
        Tensor
            A tensor containing the sum along the specified dimension.

        """
        ctx.save_for_backward(a.shape, dim)
        return a.f.add_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute the gradient for the sum operation.

        Args:
        ----
        ctx : Context
            The context containing saved tensors from the forward pass.
        grad_output : Tensor
            The gradient of the output tensor.

        Returns:
        -------
        Tuple[Tensor, float]
            The gradient of the input tensor and a placeholder float.

        """
        (a_shape, dim) = ctx.saved_values
        return grad_output, 0.0


class LT(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Compute the element-wise 'less than' comparison of two tensors.

        Args:
        ----
        ctx : Context
            The context for saving tensors for the backward pass.
        a : Tensor
            The first input tensor.
        b : Tensor
            The second input tensor.

        Returns:
        -------
        Tensor
            A boolean tensor containing the result of the 'less than' comparison.

        """
        return a.f.lt_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradient for the 'less than' operation.

        Args:
        ----
        ctx : Context
            The context containing saved tensors from the forward pass.
        grad_output : Tensor
            The gradient of the output tensor.

        Returns:
        -------
        Tuple[Tensor, Tensor]
            The gradients of the input tensors (both zero).

        """
        return zeros(grad_output.shape), zeros(grad_output.shape)


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Compute the element-wise equality comparison of two tensors.

        Args:
        ----
        ctx : Context
            The context for saving tensors for the backward pass.
        a : Tensor
            The first input tensor.
        b : Tensor
            The second input tensor.

        Returns:
        -------
        Tensor
            A boolean tensor containing the result of the equality comparison.

        """
        ctx.save_for_backward(a, b)
        return a.f.eq_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradient for the equality operation.

        Args:
        ----
        ctx : Context
            The context containing saved tensors from the forward pass.
        grad_output : Tensor
            The gradient of the output tensor.

        Returns:
        -------
        Tuple[Tensor, Tensor]
            The gradients of the input tensors (both zero).

        """
        return zeros(grad_output.shape), zeros(grad_output.shape)


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        """Check if two tensors are element-wise close to each other.

        Args:
        ----
        ctx : Context
            The context for saving tensors for the backward pass.
        a : Tensor
            The first input tensor.
        b : Tensor
            The second input tensor.

        Returns:
        -------
        Tensor
            A boolean tensor indicating where elements of a and b are close.

        """
        ctx.save_for_backward(a)
        return a.f.is_close_zip(a, b)


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tensor) -> Tensor:
        """Permute the dimensions of the input tensor.

        Args:
        ----
        ctx : Context
            The context for saving tensors for the backward pass.
        a : Tensor
            The input tensor to be permuted.
        order : Tensor
            A tensor specifying the new order of dimensions.

        Returns:
        -------
        Tensor
            The permuted tensor.

        """
        rev_order = np.array([0 for _ in range(len(order._tensor._storage))])

        for i, j in enumerate(order._tensor._storage):
            rev_order[int(j)] = i

        ctx.save_for_backward(rev_order)

        return minitorch.Tensor(
            a._tensor.permute(*order._tensor._storage), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute the gradient for the permute operation.

        Args:
        ----
        ctx : Context
            The context containing saved tensors from the forward pass.
        grad_output : Tensor
            The gradient of the output tensor.

        Returns:
        -------
        Tuple[Tensor, float]
            The gradient of the input tensor and a placeholder float.

        """
        (rev_order,) = ctx.saved_values

        return (
            minitorch.Tensor(
                grad_output._tensor.permute(*rev_order), backend=grad_output.backend
            ),
            0.0,
        )


##### FINISH


class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        """Reshape the input tensor to a new shape.

        Args:
        ----
        ctx : Context
            The context for saving tensors for the backward pass.
        a : Tensor
            The input tensor to be reshaped.
        shape : Tensor
            A tensor specifying the new shape.

        Returns:
        -------
        Tensor
            The reshaped tensor.

        """
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Compute the gradient for the view operation.

        Args:
        ----
        ctx : Context
            The context containing saved tensors from the forward pass.
        grad_output : Tensor
            The gradient of the output tensor.

        Returns:
        -------
        Tuple[Tensor, float]
            The gradient of the input tensor and a placeholder float.

        """
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size `shape`.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape `shape`.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:`Tensor` : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    """Compute the central difference approximation of the gradient.

    Args:
    ----
    f : Any
        The function to compute the gradient for.
    *vals : Tensor
        The input tensors to the function.
    arg : int, optional
        The index of the argument to compute the gradient for (default is 0).
    epsilon : float, optional
        The small value to use for the central difference approximation (default is 1e-6).
    ind : UserIndex
        The index within the tensor to compute the gradient for.

    Returns:
    -------
    float
        The approximated gradient value.

    """
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )
