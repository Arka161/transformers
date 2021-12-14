import torch
from torch import nn
import torch.nn.functional as F


class AllToAll(torch.autograd.Function):

  @staticmethod
  def forward(ctx, inputs, split_dimension, concat_dimension, split_count, groups=None):
    ctx.split_dimension = split_dimension
    ctx.concat_dimension = concat_dimension
    ctx.split_count = split_count
    ctx.groups = groups
    output = xm.all_to_all(inputs, split_dimension, concat_dimension, split_count, groups)
    #print(f"AllToAll forward. output.shape: {output.shape}")
    return output

  @staticmethod
  def backward(ctx, grad_outputs):
    #print(f"AllToAll backward.grad_outputs: {grad_outputs.shape}")
    return AllToAll.apply(grad_outputs, ctx.concat_dimension, ctx.split_dimension, ctx.split_count, ctx.groups), None, None, None, None

def all_to_all(input, split_dimension, concat_dimension, split_count, groups=None):
  """Performs an all-to-all distributed operation on the input tensor.
  This is the same as `xm.all_to_all()` but supports autograd differentiation.
  Args:
    input: A tensor of any dimension.
    split_dimension: The dimension to split the input tensor along.
    concat_dimension: The dimension to concatenate the output tensors along.
    split_count: The number of chunks to split the input tensor into.
    groups (list, optional): A list of list, representing the replica groups for
      the `all_to_all()` operation. Example: `[[0, 1, 2, 3], [4, 5, 6, 7]]`
        defines two groups, one with the `[0, 1, 2, 3]` replicas and one with
        the `[4, 5, 6, 7]` replicas. If `None` there will be only one group with
        all the replicas in it.
  Returns:
    The reduced value across the selected replicas.
  """
  import torch_xla
  import torch_xla.core.xla_model as xm
  import torch_xla.distributed.xla_multiprocessing as xmp
  from torch_xla.core.functions import all_reduce, all_gather
  return AllToAll.apply(input, split_dimension, concat_dimension, split_count, groups)

