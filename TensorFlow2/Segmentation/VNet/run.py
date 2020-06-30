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
import os
from time import time

import numpy as np
#from PIL import Image
import horovod.tensorflow as hvd
import tensorflow as tf

from utils.losses import partial_losses
from utils.parse_results import process_performance_stats


def restore_checkpoint(model, model_dir):
    try:
        model.load_weights(os.path.join(model_dir, "checkpoint"))
    except:
        print("Failed to load checkpoint, model will have randomly initialized weights.")
    return model


#by default run a normal evaluation
def meta_eval(iterator, n_val_samples, eval_id,params, model, dataset, logger, step = None):
    @tf.function
    def validation_step(features_eval, labels_eval):
        output_eval_map = model(features_eval, training=False)
        crossentropy_eval_loss, dice_eval_loss = partial_losses(output_eval_map, labels_eval)
        ce_eval_loss(crossentropy_eval_loss)
        f1_eval_loss(dice_eval_loss)
        
        return output_eval_map
    

    ce_eval_loss = tf.keras.metrics.Mean(name='ce_loss')
    f1_eval_loss = tf.keras.metrics.Mean(name='dice_loss')
    #for val_samples in validations_sets:
    f1_eval_loss.reset_states()
    ce_eval_loss.reset_states()

    eval_size = n_val_samples
    for iteration_eval, (id_eval_data,images_eval, labels_eval) in enumerate(iterator):
        print("id_eval_image_level", id_eval_data)
        images_eval = tf.reshape(images_eval, images_eval.shape + [1])
        #labels_eval = tf.reshape(labels_eval, labels_eval.shape + [1])
        output_eval_map = validation_step(images_eval, labels_eval)
             
        if iteration_eval >= eval_size // params.batch_size:
            break

    if dataset.eval_size > 0:
        logger.log(step=(),
                    data={"eval_ce_loss_"+eval_id: float(ce_eval_loss.result()),
                            "eval_dice_loss_"+eval_id: float(f1_eval_loss.result()),
                            "eval_total_loss_"+eval_id: float(f1_eval_loss.result() + ce_eval_loss.result()),
                            "eval_dice_score_"+eval_id: 1.0 - float(f1_eval_loss.result())})
        logger.flush()
        tensorboard_sumaries(os.path.join(params.model_dir, eval_id), images_eval, labels_eval, output_eval_map,
                             {"ce_loss":ce_eval_loss.result(),
                              "dice_loss":f1_eval_loss.result(),
                              "total_loss":f1_eval_loss.result() + ce_eval_loss.result()},
                             params.input_shape, dataset.biig_json.labels.values(),step=step)  
            
            
        

def train_and_meta_eval(params, model, dataset, logger, augment = True):
    
    #writer = tf.summary.create_file_writer("/results/tenb")
    validations_sets = [None] + dataset.biig_json._meta_validation_samples #nornmal eval + meta 
    iterators = [dataset.eval_fn(val_samples) for val_samples in validations_sets] #iterators are created just once this way so is much faster
    
    np.random.seed(params.seed)
    tf.random.set_seed(params.seed)
    max_steps = params.max_steps // hvd.size()

    optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)
    if params.use_amp:
        optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, "dynamic")

    ce_loss = tf.keras.metrics.Mean(name='ce_loss')
    f1_loss = tf.keras.metrics.Mean(name='dice_loss')

    @tf.function
    def train_step(features, labels, warmup_batch=False):
        with tf.GradientTape() as tape:
            output_map = model(features)

            crossentropy_loss, dice_loss = partial_losses(output_map, labels)
            added_losses = tf.add(crossentropy_loss, dice_loss, name="total_loss_ref")
            loss = added_losses + params.weight_decay * tf.add_n(
                [tf.nn.l2_loss(v) for v in model.trainable_variables if 'batch_normalization' not in v.name])

            if params.use_amp:
                loss = optimizer.get_scaled_loss(loss)
        tape = hvd.DistributedGradientTape(tape)
        gradients = tape.gradient(loss, model.trainable_variables)
        if params.use_amp:
            gradients = optimizer.get_unscaled_gradients(gradients)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Note: broadcast should be done after the first gradient step to ensure optimizer
        # initialization.
        if warmup_batch:
            hvd.broadcast_variables(model.variables, root_rank=0)
            hvd.broadcast_variables(optimizer.variables(), root_rank=0)

        ce_loss(crossentropy_loss)
        f1_loss(dice_loss)
        return loss, output_map

    if params.benchmark:
        assert max_steps * hvd.size() > params.warmup_steps, \
            "max_steps value has to be greater than warmup_steps"
        timestamps = np.zeros((hvd.size(), max_steps * hvd.size() + 1), dtype=np.float32)
        for iteration, (images, labels) in enumerate(dataset.train_fn(drop_remainder=True)):
            t0 = time()
            loss = train_step(images, labels, warmup_batch=iteration == 0).numpy()
            timestamps[hvd.rank(), iteration] = time() - t0
            if iteration >= max_steps * hvd.size():
                break
        timestamps = np.mean(timestamps, axis=0)
        if hvd.rank() == 0:
            throughput_imgps, latency_ms = process_performance_stats(timestamps, params)
            logger.log(step=(),
                       data={"throughput_train": throughput_imgps,
                             "latency_train": latency_ms})
    else:
        for iteration, (id_data, images, labels) in enumerate(dataset.train_fn(augment)):
            #print("")
            #print("ID DATA TRAIN", id_data)
            images = tf.reshape(images, images.shape + [1])
            #labels = tf.reshape(labels, labels.shape + [1])
            loss, output_map = train_step(images, labels, warmup_batch=iteration == 0)
            if (hvd.rank() == 0) and (iteration % params.log_every == 0):
                print("")
                print("ID DATA TRAIN", id_data)
                logger.log(step=(iteration, max_steps),
                           data={"train_ce_loss": float(ce_loss.result()),
                                 "train_dice_loss": float(f1_loss.result()),
                                 "train_total_loss": float(f1_loss.result() + ce_loss.result())})
                tensorboard_sumaries(os.path.join(params.model_dir, "train"), images, labels, output_map,
                                     {"ce_loss":ce_loss.result(),
                                      "dice_loss":f1_loss.result(), 
                                      "total_loss":f1_loss.result() + ce_loss.result()},
                                     params.input_shape, dataset.biig_json.labels.values(),step=iteration )  
                
                for i,iterator in enumerate(iterators): #iterators are created just once this way so is much faster
                    eval_id = "eval" if validations_sets[i] is None else validations_sets[i][0][0]
                    eval_size = dataset.eval_size if validations_sets[i] is None else len(validations_sets[i])
                    print("")
                    print("EVAL FIELDS", eval_id, eval_size)
                    meta_eval(iterator, eval_size, eval_id, params, model, dataset, logger, step=iteration)
                
                f1_loss.reset_states()
                ce_loss.reset_states()

            if iteration >= max_steps:
                break
        if hvd.rank() == 0:
            model.save_weights(os.path.join(params.model_dir, "checkpoint"))
    logger.flush()

    
def tensorboard_sumaries(log_dir, images, labels, output_map, scalars_dict, input_shape, labels_values, step = None):
    
    mid_image = int(input_shape[2] / 2)
    step_im = 1            
    n_feats = (images[0, :, :, mid_image:mid_image + step_im, 0:1] + np.abs(np.min(images)) )/(np.max(images) + np.abs(np.min(images)))
    n_labels = 85*tf.cast(labels[0, :, :, mid_image:mid_image + step_im], dtype=tf.uint8)
    n_labels = tf.expand_dims(n_labels, -1)
    
    writer = tf.summary.create_file_writer(log_dir)       
    #Add summaries for tensorboard
    with writer.as_default():
        [tf.summary.scalar(k, float(scalars_dict[k]), step=step) for k in scalars_dict.keys()]
        
        tf.summary.image("Image", tf.transpose(n_feats, [2, 1, 0, 3]), max_outputs=3, step=step)
        print("SHAPES", tf.shape(n_labels), tf.shape(n_feats))
        tf.summary.image("Mask", tf.transpose(n_labels, [2, 1, 0, 3]), max_outputs=3, step=step)
        [tf.summary.image(str(i)+"_"+val, tf.transpose(output_map[0, :, :, mid_image:mid_image + step_im, i:(i + 1)], [2, 1, 0, 3]), max_outputs=3, step=step)
         for i,val in enumerate(labels_values)]
                    
        writer.flush()
    

def train(params, model, dataset, logger, augment = True):
    np.random.seed(params.seed)
    tf.random.set_seed(params.seed)
    max_steps = params.max_steps // hvd.size()

    optimizer = tf.keras.optimizers.Adam(learning_rate=params.learning_rate)
    if params.use_amp:
        optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(optimizer, "dynamic")

    ce_loss = tf.keras.metrics.Mean(name='ce_loss')
    f1_loss = tf.keras.metrics.Mean(name='dice_loss')

    @tf.function
    def train_step(features, labels, warmup_batch=False):
        with tf.GradientTape() as tape:
            output_map = model(features)

            crossentropy_loss, dice_loss = partial_losses(output_map, labels)
            added_losses = tf.add(crossentropy_loss, dice_loss, name="total_loss_ref")
            loss = added_losses + params.weight_decay * tf.add_n(
                [tf.nn.l2_loss(v) for v in model.trainable_variables
                 if 'batch_normalization' not in v.name])

            if params.use_amp:
                loss = optimizer.get_scaled_loss(loss)
        tape = hvd.DistributedGradientTape(tape)
        gradients = tape.gradient(loss, model.trainable_variables)
        if params.use_amp:
            gradients = optimizer.get_unscaled_gradients(gradients)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Note: broadcast should be done after the first gradient step to ensure optimizer
        # initialization.
        if warmup_batch:
            hvd.broadcast_variables(model.variables, root_rank=0)
            hvd.broadcast_variables(optimizer.variables(), root_rank=0)

        ce_loss(crossentropy_loss)
        f1_loss(dice_loss)
        return loss, output_map

    if params.benchmark:
        assert max_steps * hvd.size() > params.warmup_steps, \
            "max_steps value has to be greater than warmup_steps"
        timestamps = np.zeros((hvd.size(), max_steps * hvd.size() + 1), dtype=np.float32)
        for iteration, (images, labels) in enumerate(dataset.train_fn(drop_remainder=True)):
            t0 = time()
            loss = train_step(images, labels, warmup_batch=iteration == 0).numpy()
            timestamps[hvd.rank(), iteration] = time() - t0
            if iteration >= max_steps * hvd.size():
                break
        timestamps = np.mean(timestamps, axis=0)
        if hvd.rank() == 0:
            throughput_imgps, latency_ms = process_performance_stats(timestamps, params)
            logger.log(step=(),
                       data={"throughput_train": throughput_imgps,
                             "latency_train": latency_ms})
    else:
        for iteration, (id_data, images, labels) in enumerate(dataset.train_fn(augment)):
            images = tf.reshape(images, images.shape + [1])
            labels = tf.reshape(labels, labels.shape + [1])
            loss, output_map = train_step(images, labels, warmup_batch=iteration == 0)
            if (hvd.rank() == 0) and (iteration % params.log_every == 0):
                logger.log(step=(iteration, max_steps),
                           data={"train_ce_loss": float(ce_loss.result()),
                                 "train_dice_loss": float(f1_loss.result()),
                                 "train_total_loss": float(f1_loss.result() + ce_loss.result())})
                
            
                f1_loss.reset_states()
                ce_loss.reset_states()

            if iteration >= max_steps:
                break
        if hvd.rank() == 0:
            model.save_weights(os.path.join(params.model_dir, "checkpoint"))
    logger.flush()


def evaluate(params, model, dataset, logger):
    ce_loss = tf.keras.metrics.Mean(name='ce_loss')
    f1_loss = tf.keras.metrics.Mean(name='dice_loss')

    @tf.function
    def validation_step(features, labels):
        output_map = model(features, training=False)
        crossentropy_loss, dice_loss = partial_losses(output_map, labels)
        ce_loss(crossentropy_loss)
        f1_loss(dice_loss)

    for iteration, (id_data,images, labels) in enumerate(dataset.eval_fn()):
        images = tf.reshape(images, images.shape + [1])
        labels = tf.reshape(labels, labels.shape + [1])
        validation_step(images, labels)
        if iteration >= dataset.eval_size // params.batch_size:
            break
    if dataset.eval_size > 0:
        logger.log(step=(),
                   data={"eval_ce_loss": float(ce_loss.result()),
                         "eval_dice_loss": float(f1_loss.result()),
                         "eval_total_loss": float(f1_loss.result() + ce_loss.result()),
                         "eval_dice_score": 1.0 - float(f1_loss.result())})

    logger.flush()


def predict(params, model, dataset, logger):

    @tf.function
    def prediction_step(features):
        return model(features, training=False)

    if params.benchmark:
        assert params.max_steps > params.warmup_steps, \
            "max_steps value has to be greater than warmup_steps"
        timestamps = np.zeros(params.max_steps + 1, dtype=np.float32)
        for iteration, images in enumerate(dataset.test_fn(count=None, drop_remainder=True)):
            t0 = time()
            prediction_step(images)
            timestamps[iteration] = time() - t0
            if iteration >= params.max_steps:
                break
        throughput_imgps, latency_ms = process_performance_stats(timestamps, params)
        logger.log(step=(),
                   data={"throughput_test": throughput_imgps,
                         "latency_test": latency_ms})
    else:
        predictions = np.concatenate([prediction_step(images).numpy()
                                      for images in dataset.test_fn(count=1)], axis=0)
        binary_masks = [np.argmax(p, axis=-1).astype(np.uint8) * 255 for p in predictions]
        multipage_tif = [Image.fromarray(mask).resize(size=(512, 512), resample=Image.BILINEAR)
                         for mask in binary_masks]

        output_dir = os.path.join(params.model_dir, 'predictions')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        multipage_tif[0].save(os.path.join(output_dir, 'test-masks.tif'),
                              compression="tiff_deflate",
                              save_all=True,
                              append_images=multipage_tif[1:])
    logger.flush()
