# Copyright 2014-2019 The ODL contributors
#
# This file is part of ODL.
#
# This Source Code Form is subject to the terms of the Mozilla Public License,
# v. 2.0. If a copy of the MPL was not distributed with this file, You can
# obtain one at https://mozilla.org/MPL/2.0/.

"""Utilities for converting ODL operators to pytorch layers.

This requires the ``torch`` module from the ``pytorch`` package,
see `the pytorch installation guide
<https://github.com/pytorch/pytorch#installation>`_ for instructions.
"""

from __future__ import division

import warnings

import numpy as np
import torch
from packaging.version import parse as parse_version

from odl import Operator

if parse_version(torch.__version__) < parse_version('0.4'):
    warnings.warn("This interface is designed to work with Pytorch >= 0.4",
                  RuntimeWarning, stacklevel=2)

__all__ = ('OperatorFunction', 'OperatorModule')


class OperatorFunction(torch.autograd.Function):

    """Wrapper of an ODL operator as a ``torch.autograd.Function``.

    This wrapper exposes an `Operator` object to the PyTorch autograd
    machinery by implementing custom ``forward()`` and ``backward()``
    methods.

    These methods should not be used directly. Instead, in a ``Module``,
    the call ``OperatorFunction.apply(operator, input_tensor)`` will
    apply the ``forward()`` method correctly and register gradients
    for the ``backward()`` step during backpropagation.

    The application of ``op`` to multiple inputs is done automatically
    in the background. The only requirement is that the shape of an
    input *ends with* the input shape that ``op`` expects, see below.

    """

    @staticmethod
    def forward(ctx, operator, input):
        """Evaluate forward pass on the input.

        Parameters
        ----------
        ctx : context object
            Object to communicate information between forward and backward
            passes.
        operator : `Operator`
            ODL operator to be wrapped. For gradient computations to
            work, ``operator.derivative(x).adjoint`` must be implemented.
        input : `torch.Tensor`
            Point at which to evaluate the operator.

        Returns
        -------
        result : `torch.Tensor`
            Tensor holding the result of the evaluation.
        """
        if not isinstance(operator, Operator):
            raise TypeError(
                "`operator` must be an `Operator` instance, got {!r}"
                "".format(operator)
            )

        # Save operator for backward; input only needs to be saved if
        # the operator is nonlinear (for `operator.derivative(input)`)
        ctx.operator = operator

        if not operator.is_linear:
            # Only needed for nonlinear operators
            ctx.save_for_backward(input)

        # TODO(kohr-h): use GPU memory directly when possible
        # TODO(kohr-h): remove `copy_if_zero_strides` when NumPy 1.16.0
        # is required
        input_arr = copy_if_zero_strides(input.cpu().detach().numpy())

        # Determine how to loop over extra shape "left" of the operator
        # domain shape
        in_shape = input_arr.shape
        op_in_shape = operator.domain.shape
        if operator.is_functional:
            op_out_shape = ()
            op_out_dtype = operator.domain.dtype
        else:
            op_out_shape = operator.range.shape
            op_out_dtype = operator.range.dtype

        extra_shape = in_shape[:-len(op_in_shape)]
        if in_shape[-len(op_in_shape):] != op_in_shape:
            shp_str = str(op_in_shape).strip('(,)')
            raise ValueError(
                'input tensor has wrong shape: expected (*, {}), got {}'
                ''.format(shp_str, in_shape)
            )

        # Store some information on the context object
        ctx.op_in_shape = op_in_shape
        ctx.op_out_shape = op_out_shape
        ctx.extra_shape = extra_shape
        ctx.op_in_dtype = operator.domain.dtype
        ctx.op_out_dtype = op_out_dtype

        # Evaluate the operator on all inputs in a loop
        if extra_shape:
            # Multiple inputs: flatten extra axes, then do one entry at a time
            input_arr_flat_extra = input_arr.reshape((-1,) + op_in_shape)
            results = []
            for inp in input_arr_flat_extra:
                results.append(operator(inp))

            # Stack results, reshape to the expected output shape and enforce
            # correct dtype
            result_arr = np.stack(results).astype(op_out_dtype, copy=False)
            result_arr = result_arr.reshape(extra_shape + op_out_shape)
        else:
            # Single input: evaluate directly
            result_arr = np.asarray(
                operator(input_arr)
            ).astype(op_out_dtype, copy=False)

        # Convert back to tensor
        tensor = torch.from_numpy(result_arr).to(input.device)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        r"""Apply the adjoint of the derivative at ``grad_output``.

        This method is usually not called explicitly but as a part of the
        ``backward()`` pass of a backpropagation step.

        Parameters
        ----------
        ctx : context object
            Object to communicate information between forward and backward
            passes.
        grad_output : `torch.Tensor`
            Tensor to which the Jacobian should be applied. See Notes
            for details.

        Returns
        -------
        gradients : tuple
            Tuple ``(None, grad_input)``, where the ``None`` part is due to
            the first argument of ``forward`` being the ODL operator that
            does not require a gradient. The ``grad_input`` tensor is the
            result of applying the Jacobian to ``grad_output``.
            See Notes for details.

        Notes
        -----
        This method applies the contribution of this node, i.e., the
        transpose of the Jacobian of its outputs with respect to its inputs,
        to the gradients of some cost function with respect to the outputs
        of this node.

        Hence, the parameter ``grad_output`` is a tensor containing
        :math:`y = \nabla C(f(x))`. Then, ``backward`` boils down to
        computing ``[f'(x)^*(y)]`` using the input ``x`` stored during
        the previous `forward` pass.
        """
        # Return early if there's nothing to do
        if not ctx.needs_input_grad[1]:
            return None, None

        operator = ctx.operator

        # Get `operator` and `input` from the context object (the input
        # is only needed for nonlinear operators)
        if not operator.is_linear:
            # TODO: implement directly for GPU data
            # TODO(kohr-h): remove `copy_if_zero_strides` when NumPy 1.16.0
            # is required
            input_arr = copy_if_zero_strides(
                ctx.saved_tensors[0].detach().cpu().numpy()
            )

        # ODL weights spaces, pytorch doesn't, so we need to handle this
        try:
            dom_weight = operator.domain.weighting.const
        except AttributeError:
            dom_weight = 1.0
        try:
            ran_weight = operator.range.weighting.const
        except AttributeError:
            ran_weight = 1.0
        scaling = dom_weight / ran_weight

        # Convert `grad_output` to NumPy array
        grad_output_arr = copy_if_zero_strides(
            grad_output.detach().cpu().numpy()
        )

        # Get shape information from the context object
        op_in_shape = ctx.op_in_shape
        op_out_shape = ctx.op_out_shape
        extra_shape = ctx.extra_shape
        op_in_dtype = ctx.op_in_dtype

        # Check if `grad_output` is consistent with `extra_shape` and
        # `op_out_shape`
        if grad_output_arr.shape != extra_shape + op_out_shape:
            raise ValueError(
                'expected tensor of shape {}, got shape {}'
                ''.format(extra_shape + op_out_shape, grad_output_arr.shape)
            )

        # Evaluate the (derivative) adjoint on all inputs in a loop
        if extra_shape:
            # Multiple gradients: flatten extra axes, then do one entry
            # at a time
            grad_output_arr_flat_extra = grad_output_arr.reshape(
                (-1,) + op_out_shape
            )

            results = []
            if operator.is_linear:
                for ograd in grad_output_arr_flat_extra:
                    results.append(np.asarray(operator.adjoint(ograd)))
            else:
                # Need inputs, flattened in the same way as the gradients
                input_arr_flat_extra = input_arr.reshape((-1,) + op_in_shape)
                for ograd, inp in zip(
                    grad_output_arr_flat_extra, input_arr_flat_extra
                ):
                    results.append(
                        np.asarray(operator.derivative(inp).adjoint(ograd))
                    )

            # Stack results, reshape to the expected output shape and enforce
            # correct dtype
            result_arr = np.stack(results).astype(op_in_dtype, copy=False)
            result_arr = result_arr.reshape(extra_shape + op_in_shape)
        else:
            # Single gradient: evaluate directly
            if operator.is_linear:
                result_arr = np.asarray(
                    operator.adjoint(grad_output_arr)
                ).astype(op_in_dtype, copy=False)
            else:
                result_arr = np.asarray(
                    operator.derivative(input_arr).adjoint(grad_output_arr)
                ).astype(op_in_dtype, copy=False)

        # Apply scaling, convert to tensor and return
        if scaling != 1.0:
            result_arr *= scaling
        grad_input = torch.from_numpy(result_arr).to(grad_output.device)
        return None, grad_input  # return `None` for the `operator` part


class OperatorModule(torch.nn.Module):

    """Wrapper of an ODL operator as a ``torch.nn.Module``.

    This wrapper can be used as a layer in ``pytorch`` Neural Networks.
    It works with arbitrary batches and channels and supports
    backpropagation.

    Parameters
    ----------
    operator : `Operator`
        The ODL operator to be wrapped. For gradient computations to work,
        ``operator.derivative(x).adjoint`` must be implemented.


    Backpropagation works autmatically by means of the
    ``operator.derivative(x).adjoint`` machinery. To trigger it, the
    input tensor must be marked as requiring gradient:

    >>> x = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    >>> loss = op_mod(x).sum()
    >>> loss
    tensor(6., grad_fn=<SumBackward0>)
    >>> loss.backward()
    >>> x.grad
    tensor([[1., 1., 1.]])
    """

    def __init__(self, operator: Operator):
        """Initialize a new instance."""
        super(OperatorModule, self).__init__()
        self.operator = operator
        self.op_in_shape = self.operator.domain.shape
        self.op_name = self.operator.__class__.__name__
        self.op_out_shape = self.operator.range.shape

    def forward(self, x: torch.Tensor):
        """Compute forward-pass of this module on ``x``.

        Parameters
        ----------
        x : `torch.Tensor`
            Input of this layer. The contained tensor must have shape
            ``extra_shape + operator.domain.shape``, and
            ``len(extra_shape)`` must be at least 1 (batch axis).

        Returns
        -------
        out : `torch.Tensor`
            The computed output. Its tensor will have shape
            ``extra_shape + operator.range.shape``, where ``extra_shape``
            are the extra axes of ``x``.

        """
        in_shape = x.shape
        in_ndim = len(in_shape)
        op_in_shape = self.op_in_shape
        op_in_ndim = len(op_in_shape)
        if in_ndim <= op_in_ndim or in_shape[-op_in_ndim:] != op_in_shape:
            shp_str = str(op_in_shape).strip('()')
            raise ValueError(
                'input tensor has wrong shape: expected (N, *, {}), got {}'
                ''.format(shp_str, in_shape)
            )
        return OperatorFunction.apply(self.operator, x)

    def __repr__(self):
        """Return ``repr(self)``."""
        op_name = self.op_name
        op_in_shape = self.op_in_shape
        if len(op_in_shape) == 1:
            op_in_shape = op_in_shape[0]
        op_out_shape = self.op_out_shape
        if len(op_out_shape) == 1:
            op_out_shape = op_out_shape[0]

        return '{}({}) ({} -> {})'.format(
            self.__class__.__name__, op_name, op_in_shape, op_out_shape
        )


def copy_if_zero_strides(arr):
    """Workaround for NumPy issue #9165 with 0 in arr.strides."""
    assert isinstance(arr, np.ndarray)
    return arr.copy() if 0 in arr.strides else arr


if __name__ == '__main__':
    from odl.util.testutils import run_doctests
    import odl
    from torch import autograd, nn
    run_doctests(extraglobs={'np': np, 'odl': odl, 'torch': torch,
                             'nn': nn, 'autograd': autograd})
