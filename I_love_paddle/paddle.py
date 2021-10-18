import numpy as np
import paddle


def pd_max(a: paddle.Tensor, axis=0, keepdim=True):
    max_ = a.max(axis).unsqueeze(-1)
    index = paddle.argmax(a, axis=axis, keepdim=keepdim)
    # index = paddle.argmax(a, axis=axis, keepdim=keepdim)[-1].flatten()
    return max_, index
