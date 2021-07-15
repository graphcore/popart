# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified by Graphcore Ltd.
# ==============================================================================
import numpy as np


# Function taken from the original TensorFlow repository and modified to include
# weight decay in the calculation.
def rmsprop_update_numpy(
    var,
    g,
    mg,
    rms,
    mom,
    lr,
    decay=0.9,
    momentum=0.0,
    weight_decay=0.0,
    # 'L2' or 'decay'
    weight_decay_mode='L2',
    epsilon=1e-10,
    centered=False):
    if weight_decay > 0.0 and weight_decay_mode == 'L2':
        g = g + weight_decay * var
    rms_t = rms * decay + (1 - decay) * g * g
    denom_t = rms_t + epsilon
    if centered:
        mg_t = mg * decay + (1 - decay) * g
        denom_t -= mg_t * mg_t
    else:
        mg_t = mg
    mom_t = momentum * mom + lr * g / np.sqrt(denom_t, dtype=denom_t.dtype)
    if weight_decay > 0.0 and weight_decay_mode == 'decay':
        if momentum > 0.0:
            var_t = var - (weight_decay * var + mom_t)
        else:
            var_t = var - (lr * weight_decay * var + mom_t)
    else:
        var_t = var - mom_t
    return var_t, mg_t, rms_t, mom_t
