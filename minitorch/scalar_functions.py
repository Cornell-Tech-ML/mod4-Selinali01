from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple.

    Args:
    ----
        x: A float or tuple of floats.

    Returns:
    -------
        A tuple containing the input value(s).

    """
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        """Wrapper for the backward pass.

        Args:
        ----
            ctx: The context containing saved values from the forward pass.
            d_out: The derivative of the output.

        Returns:
        -------
            A tuple of gradients for each input.

        """
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        """Wrapper for the forward pass.

        Args:
        ----
            ctx: The context for saving values for the backward pass.
            *inps: Input values for the function.

        Returns:
        -------
            The result of the forward computation.

        """
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the function to the given values.

        Args:
        ----
            *vals: Input values for the function.

        Returns:
        -------
            A Scalar containing the result of the computation.

        """
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the forward pass for addition.

        Args:
        ----
            ctx: The context (unused in this function).
            a: First input value.
            b: Second input value.

        Returns:
        -------
            The sum of a and b.

        """
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the backward pass for addition.

        Args:
        ----
            ctx: The context (unused in this function).
            d_output: The derivative of the output.

        Returns:
        -------
            A tuple containing the gradients for both inputs.

        """
        return d_output, d_output


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass for the natural logarithm.

        Args:
        ----
            ctx: The context for saving values for the backward pass.
            a: Input value.

        Returns:
        -------
            The natural logarithm of a.

        """
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass for the natural logarithm.

        Args:
        ----
            ctx: The context containing saved values from the forward pass.
            d_output: The derivative of the output.

        Returns:
        -------
            The gradient for the input.

        """
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


# TODO: Implement for Task 1.2.


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the forward pass for multiplication.

        Args:
        ----
            ctx: The context for saving values for the backward pass.
            a: First input value.
            b: Second input value.

        Returns:
        -------
            The product of a and b.

        """
        ctx.save_for_backward(a, b)
        c = a * b
        return c

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the backward pass for multiplication.

        Args:
        ----
            ctx: The context containing saved values from the forward pass.
            d_output: The derivative of the output.

        Returns:
        -------
            A tuple containing the gradients for both inputs.

        """
        a, b = ctx.saved_values
        return b * d_output, a * d_output


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1/x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass for the inverse function.

        Args:
        ----
            ctx: The context for saving values for the backward pass.
            a: Input value.

        Returns:
        -------
            The inverse of a.

        """
        ctx.save_for_backward(a)
        return operators.inv(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass for the inverse function.

        Args:
        ----
            ctx: The context containing saved values from the forward pass.
            d_output: The derivative of the output.

        Returns:
        -------
            The gradient for the input.

        """
        (a,) = ctx.saved_values
        return operators.inv_back(a, d_output)


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass for negation.

        Args:
        ----
            ctx: The context (unused in this function).
            a: Input value.

        Returns:
        -------
            The negation of a.

        """
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass for negation.

        Args:
        ----
            ctx: The context (unused in this function).
            d_output: The derivative of the output.

        Returns:
        -------
            The gradient for the input.

        """
        return -d_output


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1 / (1 + e^{-x})$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass for the sigmoid function.

        Args:
        ----
            ctx: The context for saving values for the backward pass.
            a: Input value.

        Returns:
        -------
            The sigmoid of a.

        """
        out = operators.sigmoid(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass for the sigmoid function.

        Args:
        ----
            ctx: The context containing saved values from the forward pass.
            d_output: The derivative of the output.

        Returns:
        -------
            The gradient for the input.

        """
        sigma: float = ctx.saved_values[0]
        return sigma * (1.0 - sigma) * d_output


class ReLU(ScalarFunction):
    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass for the ReLU function.

        Args:
        ----
            ctx: The context containing saved values from the forward pass.
            a: The input value.

        Returns:
        -------
            The result of the ReLU operation.

        """
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass for the ReLU function.

        Args:
        ----
            ctx: The context containing saved values from the forward pass.
            d_output: The derivative of the output.

        Returns:
        -------
            The gradient for the input.

        """
        (a,) = ctx.saved_values  # Unpack the single saved value
        return operators.relu_back(a, d_output)


class Exp(ScalarFunction):
    """Exponential function f(x) = e^x."""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the forward pass for the exponential function.

        Args:
        ----
            ctx: The context for saving values for the backward pass.
            a: Input value.

        Returns:
        -------
            The exponential of a.

        """
        out = operators.exp(a)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the backward pass for the exponential function.

        Args:
        ----
            ctx: The context containing saved values from the forward pass.
            d_output: The derivative of the output.

        Returns:
        -------
            The gradient for the input.

        """
        out: float = ctx.saved_values[0]
        return d_output * out


class LT(ScalarFunction):
    """Less than function f(x, y) = 1 if x < y else 0."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the forward pass for the less than function.

        Args:
        ----
            ctx: The context (unused in this function).
            a: First input value.
            b: Second input value.

        Returns:
        -------
            1.0 if a < b, else 0.0.

        """
        return 1.0 if a < b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the backward pass for the less than function.

        Args:
        ----
            ctx: The context (unused in this function).
            d_output: The derivative of the output.

        Returns:
        -------
            A tuple of zeros as gradients (function is non-differentiable).

        """
        return 0.0, 0.0


class EQ(ScalarFunction):
    """Equal function f(x, y) = 1 if x == y else 0."""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the forward pass for the equal function.

        Args:
        ----
            ctx: The context (unused in this function).
            a: First input value.
            b: Second input value.

        Returns:
        -------
            1.0 if a == b, else 0.0.

        """
        return 1.0 if a == b else 0.0

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, float]:
        """Compute the backward pass for the equal function.

        Args:
        ----
            ctx: The context (unused in this function).
            d_output: The derivative of the output.

        Returns:
        -------
            A tuple of zeros as gradients (function is non-differentiable).

        """
        return 0.0, 0.0
