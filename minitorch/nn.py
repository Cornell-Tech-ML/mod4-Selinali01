from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    # Calculate output dimensions
    new_height = height // kh
    new_width = width // kw

    # Step 1: Split width dimension into new_width and kw
    # Shape: (batch, channel, height, new_width, kw)
    width_split = input.contiguous().view(batch, channel, height, new_width, kw)

    # Step 2: Reorder dimensions to prepare for height splitting
    # Shape: (batch, channel, new_width, height, kw)
    dimension_reordered = width_split.permute(0, 1, 3, 2, 4)

    # Step 3: Split height dimension and combine kernel dimensions
    # Shape: (batch, channel, new_width, new_height, kh * kw)
    windows = dimension_reordered.contiguous().view(
        batch, channel, new_width, new_height, kh * kw
    )

    return windows, new_height, new_width


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average 2D pooling

    Args:
    ----
        input : batch x channel x height x width
        kernel : height x width of pooling

    Returns:
    -------
        Pooled tensor

    """
    batch, channel, height, width = input.shape
    input, new_height, new_width = tile(input, kernel)
    pooled = input.mean(-1)  # Take mean over the pooling window dimension
    return pooled.view(
        batch, channel, new_height, new_width
    )  # Reshape to output dimensions


# TODO: Implement for Task 4.4.
max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Computes the argmax as a one-hot encoded tensor along the specified dimension.

    Args:
    ----
        input: Input tensor to compute argmax over
        dim: Dimension along which to find maximum values

    Returns:
    -------
        Tensor: One-hot encoded tensor with 1.0 at maximum positions

    """
    return max_reduce(input, dim) == input


class Max(Function):
    """Function implementing the max reduction operation with gradient support."""

    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Computes the maximum values along the specified dimension.

        Args:
        ----
            ctx: Context for saving values needed in backward pass
            input: Input tensor
            dim: Dimension to reduce over (as a single-element tensor)

        Returns:
        -------
            Tensor: Tensor containing maximum values along specified dimension

        """
        # Save input and dimension for backward pass
        ctx.save_for_backward(input, int(dim.item()))
        return max_reduce(input, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Computes gradient of max operation using argmax.

        Args:
        ----
            ctx (Context): Context containing saved tensors from forward pass
            grad_output (Tensor): Gradient with respect to output

        Returns:
        -------
            Tuple[Tensor, float]: Tuple of (gradient with respect to input, gradient with respect to dimension)

        """
        input, dim = ctx.saved_values
        # Gradient is only propagated through maximum elements
        return grad_output * argmax(input, dim), 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Computes maximum values along specified dimension.

    Wrapper around Max.apply that converts dimension to tensor.

    Args:
    ----
        input (Tensor): Input tensor
        dim (int): Dimension to reduce over

    Returns:
    -------
        Tensor: Maximum values along specified dimension

    """
    return Max.apply(input, tensor([dim]))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Applies softmax normalization along specified dimension.

    Args:
    ----
        input (Tensor): Input tensor
        dim (int): Dimension along which to apply softmax

    Returns:
    -------
        Tensor: Softmax probabilities (sum to 1 along dim)

    """
    # Subtract max for numerical stability
    max_vals = max_reduce(input, dim)
    shifted = input - max_vals
    exp_vals = shifted.exp()
    sum_exp = exp_vals.sum(dim=dim)
    return exp_vals / sum_exp


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Applies log softmax along specified dimension using numerically stable computation.

    Computes as x_i - max(x) - log(sum(exp(x_j - max(x)))) using the LogSumExp trick
    for numerical stability.

    Args:
    ----
        input (Tensor): Input tensor
        dim (int): Dimension along which to apply log softmax

    Returns:
    -------
        Tensor: Log of softmax probabilities

    """
    # Use logsumexp trick
    max_val = max_reduce(input, dim)
    shifted_input = input - max_val
    exps = shifted_input.exp()
    exps_sum = exps.sum(dim=dim)
    log_exps_sum = exps_sum.log()
    return shifted_input - log_exps_sum


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Applies 2D max pooling over a 4D input tensor.

    First tiles input tensor into pooling windows, then reduces each window
    by taking the maximum value.

    Args:
    ----
        input (Tensor): Input tensor of shape (batch x channel x height x width)
        kernel (Tuple[int, int]): Size of pooling window as (kernel_height, kernel_width)

    Returns:
    -------
        Tensor: Pooled tensor with reduced height and width dimensions

    """
    batch, channel, height, width = input.shape
    # Reshape input into pooling windows
    tiled_input, new_height, new_width = tile(input, kernel)
    # Take max over pooling window dimension
    pooled = max(tiled_input, len(tiled_input.shape) - 1)
    return pooled.view(batch, channel, new_height, new_width)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """Applies dropout regularization to input tensor.

    Args:
    ----
        input (Tensor): Input tensor
        rate (float): Dropout probability in range [0, 1)
        ignore (bool): If True, return input unchanged (useful for inference)

    Returns:
    -------
        Tensor: Tensor with random elements dropped out and appropriately scaled

    """
    if ignore:
        return input
    else:
        return input * (rand(input.shape) > rate)
