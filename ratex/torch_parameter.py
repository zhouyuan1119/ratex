# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

# pylint: disable=protected-access, c-extension-no-member
"""Hooks to PyTorch."""
import torch

import _RATEXC


def _to(self, *args, **kwargs):
    # Special handling converting meta tensor to lazy tensor
    if self.device.type == "meta" and isinstance(args[0], torch.device) and args[0].type == "lazy":
        ret = _RATEXC._raf_fake_parameter(self)
        return ret
    # All other cases are the same as before
    else:
        ret = super(torch.nn.parameter.Parameter, self).to(*args, **kwargs)
        if str(ret.device.type) == "lazy":
            return _RATEXC._raf_mark_parameter(ret)
        return ret

torch.nn.parameter.Parameter.to = _to
