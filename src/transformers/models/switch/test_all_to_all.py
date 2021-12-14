import sys
import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.core.functions as xf

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
    slots_per_device = 4
    size = slots_per_device * xm.xrt_world_size()
    ordinal = xm.get_ordinal()
    value = torch.tensor([ordinal] * size, dtype=torch.int32, device=device)
    result_tensor = xf.all_to_all(
        value,
        split_dimension=0,
        concat_dimension=0,
        split_count=xm.xrt_world_size())

    result = result_tensor.cpu().tolist()
    for i in range(0, xm.xrt_world_size()):
      expected = [i] * slots_per_device
      assert_stats(expected, result, slots_per_device, i)
  else:
    print(
        'Default device {} is not a TPU device'.format(device), file=sys.stderr)
    sys.exit(1)


if __name__ == '__main__':
  xmp.spawn(_mp_fn, args=())
