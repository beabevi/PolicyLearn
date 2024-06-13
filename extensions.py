import torch
from torch.profiler import record_function

import debug
import sparse_utils


def _vrange(starts, lengths, is_vrange):
    slices = lengths.cumsum(-1)
    output = slices.new_zeros(slices[-1])
    sparse_utils.forward(starts, slices, output, is_vrange)
    return output


def vrange(lengths, starts=None):
    if starts is None:
        starts = lengths.new_zeros(lengths.size(0))
    if lengths.device.type == "cuda":
        res = _vrange(starts, lengths, True)
        if debug.DEBUG:
            assert (res == torch_vrange(lengths, starts)).all()
    else:
        res = torch_vrange(lengths, starts)

    return res


def repeat_interleave(elems, lengths):
    if lengths.device.type == "cuda":
        res = _vrange(elems, lengths, False)
        if debug.DEBUG:
            assert (res == torch.repeat_interleave(elems, lengths)).all()
    else:
        res = torch.repeat_interleave(elems, lengths)
    return res


# From https://codereview.stackexchange.com/q/83018
@record_function("torch_vrange")
def torch_vrange(lengths, starts=None):
    """Create concatenated ranges of integers for multiple start/length

    Args:
        lengths (torch.array): lengths for each range (same length as starts)
        starts (torch.array): starts for each range
    Returns:
        torch.array: concatenated ranges

    See the following illustrative example:

        starts  = np.array([1, 3, 4, 6])
        lengths = np.array([0, 2, 3, 0])

        print vrange(starts, lengths)
        >>> [3 4 4 5 6]

    """

    # Create group counter that resets for each start/length
    lengths_cumsum = lengths.cumsum(-1)
    lengths_sum = lengths_cumsum[-1]

    buf = torch.arange(lengths_sum, device=lengths.device)
    cat_counter = buf - torch.repeat_interleave(lengths_cumsum - lengths, lengths)

    if starts is None:
        cat_start = buf.zero_()
    else:
        # Repeat start position index length times and concatenate
        cat_start = torch.repeat_interleave(starts, lengths)

    # Add group counter to group specific starts
    return cat_start + cat_counter
