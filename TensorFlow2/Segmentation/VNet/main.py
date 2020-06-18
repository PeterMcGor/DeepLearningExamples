# Based on mine, NVIDIA implemtation for TF1 https://github.com/NVIDIA/DeepLearningExamples/tree/master/TensorFlow/Segmentation/VNet
# and Miguel Monteiro https://github.com/MiguelMonteiro/VNet-Tensorflow
## Author: {Pedro M. Gordaliza}
## Copyright: Copyright {year}, {project_name}
## Credits: [{NVIDIA}]
## License: {license}
## Version: {0}.{1}.{0}
## Mmaintainer: {maintainer}
## Email: {pedro.macias.gordaliza@gmail.com}
## Status: {dev}

# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

"""Entry point of the application.

This file serves as entry point to the run of UNet for segmentation of neuronal processes.

Example:
    Training can be adjusted by modifying the arguments specified below::

        $ python main.py --exec_mode train --model_dir /dataset ...

"""


import subprocess
import sys
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install","-r", package])
        
install("/code-forked/TensorFlow2/Segmentation/VNet/requirements.txt") # workaround dont want to create requiremts for a test

import os

import horovod.tensorflow as hvd
import tensorflow as tf

from model.vnet import Vnet, vnet_fn
from run import train, evaluate, predict, restore_checkpoint
from utils.cmd_util import PARSER, _cmd_params
from utils.data_loader_ext import SegmentationLUNADataSet
from dllogger.logger import Logger, StdOutBackend, JSONStreamBackend, Verbosity


def main():
    """
    Starting point of the application
    """

    flags = PARSER.parse_args()
    params = _cmd_params(flags)

    backends = [StdOutBackend(Verbosity.VERBOSE)]
    if params.log_dir is not None:
        backends.append(JSONStreamBackend(Verbosity.VERBOSE, params.log_dir))
    logger = Logger(backends)

    # Optimization flags
    os.environ['CUDA_CACHE_DISABLE'] = '0'

    os.environ['HOROVOD_GPU_ALLREDUCE'] = 'NCCL'

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

    os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = 'data'

    os.environ['TF_ADJUST_HUE_FUSED'] = 'data'
    os.environ['TF_ADJUST_SATURATION_FUSED'] = 'data'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = 'data'

    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_AUTOTUNE_THRESHOLD'] = '2'

    hvd.init()

    if params.use_xla:
        tf.config.optimizer.set_jit(True)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    if params.use_amp:
        tf.keras.mixed_precision.experimental.set_policy('mixed_float16')
    else:
        os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'

    
    input_to_model_shape = (128, 128, 64)
    dataset = SegmentationLUNADataSet(data_dir=params.data_dir,
                                      json_path=params.json_path,
                                      training_keys=["Philips_Brilliance16_unknown", "TOSHIBA_Aquilion_V2.02ER003"], 
                                      test_keys=["TOSHIBA_Aquilion_V2.04ER001"], 
                                      meta_validation_keys=["Philips_Brilliance 64_unknown"], 
                                      interpolator = "bspline", 
                                      shuffle_before_split=False, 
                                      test_sample_fields=["image"], 
                                      dst_size=input_to_model_shape, 
                                      batch_size=params.batch_size,
                                      gpu_id=hvd.rank(),
                                      num_gpus=hvd.size(),
                                      seed=params.seed)
    
    # Build the  model
    model = Vnet(n_classes=4)
    model.model(list(input_to_model_shape)+[1]).summary(positions=[0.35,0.65,0.8,1.]) #workaround to summary the model as keras stylis

    if 'train' in params.exec_mode:
        train(params, model, dataset, logger, augment = True)

    if 'evaluate' in params.exec_mode:
        if hvd.rank() == 0:
            model = restore_checkpoint(model, params.model_dir)
            evaluate(params, model, dataset, logger)

    if 'predict' in params.exec_mode:
        if hvd.rank() == 0:
            model = restore_checkpoint(model, params.model_dir)
            predict(params, model, dataset, logger)


if __name__ == '__main__':
    
    main()
