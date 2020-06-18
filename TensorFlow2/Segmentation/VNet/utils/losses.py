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
"""Training and evaluation losses"""

import tensorflow as tf


# Class Dice coefficient averaged over batch
def dice_coef(predict, target, axis=1, eps=1e-6):
    intersection = tf.reduce_sum(input_tensor=predict * target, axis=axis)
    union = tf.reduce_sum(input_tensor=predict * predict + target * target, axis=axis)
    dice = (2. * intersection + eps) / (union + eps)
    return tf.reduce_mean(input_tensor=dice, axis=0)  # average over batch


def partial_losses(predict, target):
    n_classes = predict.shape[-1]

    #flat_logits = tf.reshape(tf.cast(predict, tf.float32), [tf.shape(input=predict)[0], -1, n_classes])
    #flat_labels = tf.reshape(target,[tf.shape(input=predict)[0], -1, n_classes])
    #one_hot_labels = tf.keras.backend.one_hot(flat_labels, n_classes)
    
    
    # Flattened logits and softmax - in FP32
    print("No LOGITS", predict)
    flat_logits = tf.reshape(predict, [tf.shape(predict)[0], -1, n_classes]) #No son exactamente logits, ya han pasado por softmax en este punto
    flat_logits = tf.cast(flat_logits, tf.float32)

    # One hot encoding
    #flat_labels = tf.keras.backend.flatten(target)
    print("PRE FLAT", target)
    flat_labels = tf.reshape(target, [tf.shape(target)[0], -1])
    #one_hot_labels = tf.one_hot(indices=flat_labels,depth=n_classes,dtype=tf.float32)
    one_hot_labels = tf.keras.backend.one_hot(flat_labels, n_classes)

    print("FLAT LOGITS", flat_logits)
    print("FLAT LABELS", flat_labels)
    print("ONE HOT", one_hot_labels)

    #crossentropy_loss = tf.reduce_mean(input_tensor=tf.keras.backend.binary_crossentropy(output=flat_logits, target=flat_labels),name='cross_loss_ref')
    
    #dice_loss = tf.reduce_mean(input_tensor=1 - dice_coef(flat_logits, one_hot_labels), name='dice_loss_ref')
    cce = tf.keras.losses.CategoricalCrossentropy()
    loss = cce(one_hot_labels, flat_logits)
    return loss, loss
    #return dice_loss, dice_loss
    #return crossentropy_loss, dice_loss
