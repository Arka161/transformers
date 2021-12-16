import sys
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from dist import all_to_all

def assert_stats(expected, result, slots_per_device, i):
    try:
        assert expected == result[i * slots_per_device:(i + 1) * slots_per_device]
    except:
        print(
            'Wrong result from core {}: {}'.format(i, result), file=sys.stderr)
        sys.exit(1)

def _mp_fn(index):
  device = xm.xla_device()
  if xm.xla_device_hw(device) == 'TPU':

    n_cores = xm.xrt_world_size()
    ordinal = xm.get_ordinal()
    d_model = 1
    expert_capacity = 2
    n_cores = 8
    n_experts = 8
    expert_inputs = ordinal * torch.ones([n_experts, n_cores, expert_capacity, d_model], dtype=torch.int32, device=device)
    if index == 1:
        print(expert_inputs.shape)

    result_tensor = all_to_all(
        expert_inputs,
        split_dimension=1,
        concat_dimension=0,
        split_count=xm.xrt_world_size())
    result_tensor *= ordinal

    result_cpu = result_tensor.cpu()
    if index == 1:
        print(result_cpu.shape)

    result_tensor = all_to_all(result_tensor, split_dimension=0, concat_dimension=1, split_count=xm.xrt_world_size())
    result_cpu = result_tensor.cpu()
    if index == 1:
        print(result_cpu.shape)
        print(result_cpu)

    for i in range(0, xm.xrt_world_size()):
      expected = [i] * slots_per_device
      assert_stats(expected, result, slots_per_device, i)
  else:
    print(
        'Default device {} is not a TPU device'.format(device), file=sys.stderr)
    sys.exit(1)


if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=())
