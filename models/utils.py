from typing import List, Dict

import math
import numpy as np
import torch

# __all__ = []


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def program_conv_filters(
    sequence_length: int,
    conv_filter_list: List[Dict],
    output_lower_bound: int,
    output_upper_bound: int,
    pad: bool = True,
    stride_to_pool_ratio: float = 1.00,
    trials: int = 5,
    class_name: str = "",
    verbose=False,
):
    # desired
    mid = (output_upper_bound + output_lower_bound) / 2.0
    in_out_ratio = float(sequence_length) / mid

    base_stride = np.power(
        in_out_ratio / np.prod([cf["kernel_size"] for cf in conv_filter_list], dtype=np.float64),
        1.0 / len(conv_filter_list),
    )

    for i in range(len(conv_filter_list)):
        cf = conv_filter_list[i]
        if i == 0 and len(conv_filter_list) > 1:
            total_stride = max(1.0, base_stride * cf["kernel_size"] * 0.7)
            cf["pool"] = max(1, round(np.sqrt(total_stride / stride_to_pool_ratio) * stride_to_pool_ratio * 0.3))
            cf["stride"] = max(1, round(total_stride / cf["pool"]))
        else:
            total_stride = max(1.0, base_stride * cf["kernel_size"])
            if stride_to_pool_ratio > 1.0:
                cf["pool"] = min(
                    max(1, round(np.sqrt(total_stride / stride_to_pool_ratio) * stride_to_pool_ratio)),
                    round(total_stride),
                )
                cf["stride"] = max(1, round(total_stride / cf["pool"]))
            else:
                cf["stride"] = min(max(1, round(np.sqrt(total_stride / stride_to_pool_ratio))), round(total_stride))
                cf["pool"] = max(1, round(total_stride / cf["stride"]))

        # cf['r'] = np.sqrt(total_stride / stride_pool_ratio)
        # cf['dilation'] = 1
        conv_filter_list[i] = cf

    success = False
    str_debug = f"\n{'-'*100}\nstarting from sequence length: {sequence_length}\n{'-'*100}\n"
    current_length = sequence_length

    for k in range(trials):
        if success:
            break

        for pivot in reversed(range(len(conv_filter_list))):
            current_length = sequence_length

            for cf in conv_filter_list:
                current_length = current_length // cf.get("pool", 1)
                str_debug += f"{cf} >> {current_length} "

                effective_kernel_size = (cf["kernel_size"] - 1) * cf.get("dilation", 1)
                both_side_pad = 2 * (cf["kernel_size"] // 2) if pad is True else 0
                current_length = (current_length + both_side_pad - effective_kernel_size - 1) // cf["stride"] + 1
                str_debug += f">> {current_length}\n"

            pool = conv_filter_list[pivot]["pool"]
            stride = conv_filter_list[pivot]["stride"]
            if current_length < output_lower_bound:
                if float(pool) / stride < stride_to_pool_ratio:
                    if stride > 1:
                        conv_filter_list[pivot]["stride"] = max(1, stride - 1)
                    else:
                        conv_filter_list[pivot]["pool"] = max(1, pool - 1)
                else:
                    if pool > 1:
                        conv_filter_list[pivot]["pool"] = max(1, pool - 1)
                    else:
                        conv_filter_list[pivot]["stride"] = max(1, stride - 1)
            elif current_length > output_upper_bound:
                if float(pool) / stride < stride_to_pool_ratio:
                    conv_filter_list[pivot]["pool"] = pool + 1
                else:
                    conv_filter_list[pivot]["stride"] = stride + 1
            else:
                str_debug += f">> Success!"
                success = True
                break

            str_debug += f">> Failed.."
            str_debug += f"\n{'-' * 100}\n"

    if verbose:
        print(str_debug)

    if not success:
        header = class_name + ", " if len(class_name) > 0 else ""
        raise RuntimeError(
            f"{header}conv1d_filter_programming() failed to determine "
            f"the proper convolution filter parameters. "
            f"The following is the recording for debug: {str_debug}"
        )

    output_length = current_length
    return output_length


def make_pool_or_not(base_pool, pool: int):
    def do_nothing(x):
        return x

    if pool == 1:
        return do_nothing
    elif pool > 1:
        return base_pool(pool)
    else:
        raise ValueError(f"make_pool_or_not(pool) receives an invalid value as input.")
