from abc import ABC, abstractmethod
import sys
import os
import warnings
import json
import inspect
import multiprocessing
from collections.abc import Iterable

import numpy as np
from scipy import stats
import SimpleITK as sitk
import horovod.tensorflow as hvd

import tensorflow as tf
from utils import side_functions


ERA4TB_json_minimum_fields = ["name", "description", "licence", "tensorImageSize", "modality", "labels"]

all_sample_fields = ["image", "label", "z", "channel", "anchors"]

segmentation_problem_ouput_types = [tf.float32, tf.int32]
segmentation_problem_output_shapes = ([128, 128, 64], [128, 128, 64])

need_complete_path =  ["image", "label"]



_SITK_INTERPOLATOR_DICT = {
    'nearest': sitk.sitkNearestNeighbor,
    'linear': sitk.sitkLinear,
    'gaussian': sitk.sitkGaussian,
    'label_gaussian': sitk.sitkLabelGaussian,
    'bspline': sitk.sitkBSpline,
    'hamming_sinc': sitk.sitkHammingWindowedSinc,
    'cosine_windowed_sinc': sitk.sitkCosineWindowedSinc,
    'welch_windowed_sinc': sitk.sitkWelchWindowedSinc,
    'lanczos_windowed_sinc': sitk.sitkLanczosWindowedSinc,
    None:None
}

def load_image(img_path, out_dims = None, z_slice=None, channel=None): # TODO wrong code
    """
    z_slice and channel must be a int, 'r' for random or None 
    """
    def find_correct_param(param, limit):
        return np.random.randint(0, limit) if param is 'r' else param
        
        
    img = sitk.ReadImage(img_path)
    im_dim = img.GetDimension()
    im_size = img.GetSize()
    if out_dims is None or out_dims == im_dim: # by default return the image as is loaded without slicing
        return img
    z_slice = find_correct_param(z_slice, im_size[2])
    if im_dim == 4:
        channel = find_correct_param(channel, im_size[3])
        if out_dims == 3:  # return a volume from 4D
            assert z_slice is None or channel is None
            return img[:,:,channel] if z_slice is None else img[:,:,z_slice]
    return img[:,:,z_slice] # return a 2D from volume



def treat_non_contiguous_labels(labels):  # one hot labels encoding cause problems
    def map_labels():
        new_labels = np.zeros_like(labels)
        for indx, un_value in enumerate(un):
            new_labels[labels == un_value] = indx
        return new_labels

    un = np.unique(labels)
    # check if the labels are positive
    positive = np.min(un) > -1
    # check if are contiguous
    contiguous = len(un) == len(np.arange(un.min(), un.max() + 1))
    if positive and contiguous:
        return labels
    else:
        return map_labels()



def sitk_to_np(sitk_img):
    # TODO try with more than 3D
    return np.transpose(sitk.GetArrayFromImage(sitk_img))

def resize_image(sitk_img,
                 dst_size=(128, 128, 64),
                 interpolator=sitk.sitkNearestNeighbor):
    reference_image = sitk.Image(dst_size, sitk_img.GetPixelIDValue())
    reference_image.SetOrigin(sitk_img.GetOrigin())
    reference_image.SetDirection(sitk_img.GetDirection())
    reference_image.SetSpacing(
        [sz * spc / nsz for nsz, sz, spc in zip(dst_size, sitk_img.GetSize(), sitk_img.GetSpacing())])
    
    return sitk.Resample(sitk_img, reference_image, sitk.Transform(3, sitk.sitkIdentity), interpolator)



class BiiGJsonParser:
    """
    Class to read the defined JSONs for the projects which must contain sets of paths to images
    and labels when are required (training and validation) under the following hierarchy
    subset_id:[{image:image_path, label:label_path}]
    """
    def __init__(self, json_path, data_dir = None, 
                 training_keys=[], meta_validation_keys=[], test_keys=[],
                 test_sample_fields= ["image"],
                 train_split = 0.8, split_seed = 42, shuffle_before_split = True):
        def __check_fields():
            min_fields = set(ERA4TB_json_minimum_fields + training_keys + meta_validation_keys + test_keys)
            if not self.json_data.keys() >= min_fields:
                sys.exit("JSON does not contain the neccesary minimum fields or id are incorrect "+ str(min_fields))
            
        assert len(training_keys) > 0 or len(meta_validation_keys) > 0 or len(test_keys) > 0, "At least one [training_keys, meta_validation_keys,test_keys] must contain data"
        self.json_path = json_path
        with open(json_path) as f:
            self.json_data = json.load(f)
            
        __check_fields() # Check if the necesary fields are included
        self.data_dir = data_dir
        self.train_split = train_split
        self.split_seed = split_seed
        self.shuffle = shuffle_before_split
        
        self._name = self.json_data.get("name")
        self._description = self.json_data.get("description")
        self._licence = self.json_data.get("licence")
        self._tensor_image_size = int(self.json_data.get("tensorImageSize")[0])
        self._labels = self.json_data.get("labels")
        self._modality = [self.json_data.get('modality')[k] for k in self.json_data.get('modality').keys()]
        # tuples with the paths and extra info depending on the problem. Always the image path, can vary for mask path, slice info, reports_annotations, achors, etc. I mantain ID cause some algorithms needs no mix of datasets at training time
        self._training_samples = self.fill_samples(training_keys, all_sample_fields)
        self._validation_samples = None
        self.make_split() # validation_samples is populated from tuples from a subset of the training samples
        self._meta_validation_samples = self.fill_samples(meta_validation_keys, all_sample_fields, group_by_id=True) # tuples with ground truth data but not from the training set (new dataset)
        self._test_samples = self.fill_samples(test_keys, test_sample_fields) # tuples without ground truth (or ground truth will be ignored)
        
    @property
    def name(self):
        return self._name
    
    @property
    def description(self):
        return self._description
    
    @property
    def licence(self):
        return self._licence
    
    @property
    def modality(self):
        return self._modality
        
    @property
    def tensor_image_size(self):
        return self._tensor_image_size
        
    @property
    def labels(self):
        return self._labels
    
    @property
    def training_samples(self):
        return self._training_samples
    
    @property
    def validation_samples(self):
        return self._validation_samples
    
    @property
    def meta_validation_samples(self):
        return self._meta_validation_samples
    
    @property
    def test_samples(self):
        return self._test_samples
        
              
    def fill_samples(self, keys_list, to_extract_fields, group_by_id = False):
        def add_parent_dir(data, sample_dict_key):
            return data if sample_dict_key not in need_complete_path else os.path.join(self.data_dir, data) if data is not None else None
        if group_by_id:
            return [[[set_id] + [add_parent_dir(sample_dict.get(k, None), k) for k in to_extract_fields] for sample_dict in self.json_data.get(set_id)] for set_id in keys_list ] if len(keys_list) > 0 else None
        return [[set_id] + [add_parent_dir(sample_dict.get(k, None), k) for k in to_extract_fields] for set_id in keys_list for sample_dict in self.json_data.get(set_id)] if len(keys_list) > 0 else None
    
    def make_split(self): # TODO this method is just correct if the split is performed at the __init__
        if self.training_samples is None:
            sys.exit("There is not training samples")
        np.random.seed(self.split_seed)
        train_size = int(len(self.training_samples) * self.train_split) + 1
        if train_size == len(self.training_samples):
            warnings.warn("Insuficient number of training samples to get a split of "+str( (1 - self.train_split) * 100)+"%")
        
        if self.shuffle:
            np.random.shuffle(self._training_samples)
        self._validation_samples = self._training_samples[train_size:] if train_size > 1 else None
        self._training_samples = self._training_samples[:train_size] 
        
    
class BiiGDataset(ABC):
    def __init__(self,
                 dst_size=segmentation_problem_output_shapes,
                 dst_type=segmentation_problem_ouput_types,
                 seed=42, 
                 batch_size=1, 
                 images_as_n_dim = 3,
                 num_gpus=1, 
                 gpu_id=0, 
                 **kwargs):
        """"
        kwargs for BiiGJsonParser
        """
        self.biig_json = BiiGJsonParser(**kwargs)
        self.real_dimension = self.biig_json.tensor_image_size
        ## Images of any real dimesion will be trated as the one pointed here. In some cases would be efficient storage image in RAM memory so this info would be required at train_fn/eval_fn/test_fn
        self.dimension = images_as_n_dim 
        self.dst_size = dst_size
        self.dst_type = dst_type
        self._seed = seed
        self._batch_size = batch_size
        
        self._num_gpus = num_gpus
        self._gpu_id = gpu_id
        
        np.random.seed(self._seed)

    def load_sample_as_images(self, sample):
        z = sample[3] 
        ch = sample[4]
        image = load_image(sample[1], out_dims=self.dimension, z_slice=z, channel=ch)
        if sample[2] is None:
            return image, None  
        else:
            return image, load_image(sample[2], out_dims=self.dimension, z_slice=z, channel=ch)
    
    def modality_normalization(self, np_image):
        if "CT" not in self.biig_json.modality:
            return stats.zscore(np_image, axis=None) # zscore--> mean 0, std 1
        if "CT" in self.biig_json.modality:
            np_image[np_image < -1000] = -1000
            np_image[np_image > 1000] = 1000
            
            
        return np_image
             
    @property
    def train_steps(self):
        global_batch_size = hvd.size() * self._batch_size

        return math.ceil(
            len(self.biig_json.training_samples) / global_batch_size)

    @property
    def eval_steps(self):
        return np.ceil(len(self.biig_json.validation_samples) / self._batch_size)
    
    @property
    def eval_size(self):
        return len(self.biig_json.validation_samples)

    @property
    def test_steps(self):
        return np.ceil(len(self.biig_json.test_samples) / self._batch_size)
    
    def preprocesing_transforms(self, samples, transforms, **kwargs):
       #bloque para cargar como itk, tranformadas itk, tranformar np, tranformadas np
        for sample in samples:
            id_dataset = sample[0]
            image, label = self.load_sample_as_images(sample)
            params = image if label is None else [image, label]
            pre_results = list(side_functions.concat_fn(transforms, image, label))
            #pre_results = pre_results if isinstance(pre_results, Iterable) else [pre_results]
    
            yield tuple([id_dataset]+pre_results)
                
        
    @abstractmethod
    def train_fn(self):
        pass
    
    @abstractmethod
    def eval_fn(self):
        pass
    
    @abstractmethod
    def test_fn():
        pass
 

#TODO remember tf.function after debugging    
class SegmentationLUNADataSet(BiiGDataset):
    def __init__(self, interpolator=None, **kwargs):
        super().__init__(**kwargs)
        self.interpolator = None
        self.set_interpolator(interpolator)
        
       
    def set_interpolator(self, interpolator):
        assert interpolator in _SITK_INTERPOLATOR_DICT.keys(), "Incorrect interpolator, must be one of the following "+str(list(_SITK_INTERPOLATOR_DICT.keys()))
        self.interpolator = _SITK_INTERPOLATOR_DICT.get(interpolator)
    
        
    
    @tf.function
    def _augment(self, id_dataset, x, y):
        # Horizontal flip
        h_flip = tf.random.uniform([]) > 0.5
        x = tf.cond(h_flip, lambda: tf.image.flip_left_right(x), lambda: x)
        y = tf.cond(h_flip, lambda: tf.image.flip_left_right(y), lambda: y)

        # Vertical flip
        v_flip = tf.random.uniform([]) > 0.5
        x = tf.cond(v_flip, lambda: tf.image.flip_up_down(x), lambda: x)
        y = tf.cond(v_flip, lambda: tf.image.flip_up_down(y), lambda: y)

        return id_dataset, x, y

        
    def train_fn(self, augment):
        def resize_sample(image, label):
            return resize_image(image, self.dst_size, self.interpolator), resize_image(label, self.dst_size, _SITK_INTERPOLATOR_DICT.get("nearest"))
        def flip_z_axes(image, label):
            do_flip = np.random.rand() > 0.5
            #print("DO FLIP", do_flip)
            return sitk.Flip(image, [False, False, do_flip]), sitk.Flip(label, [False, False, do_flip])
        
        no_tf_transforms = [resize_sample,
                            lambda x,y: (sitk_to_np(x), sitk_to_np(y)),
                            lambda x,y: (x,treat_non_contiguous_labels(y)),
                            lambda x,y: (self.modality_normalization(x), y)]
        if augment:
            no_tf_transforms = [resize_sample,
                                lambda x,y: flip_z_axes(x,y),
                                lambda x,y: (sitk_to_np(x), sitk_to_np(y)),
                                lambda x,y: (x,treat_non_contiguous_labels(y)),
                                lambda x,y: (self.modality_normalization(x), y)]
        
        dataset = tf.data.Dataset.from_generator(lambda:self.preprocesing_transforms(self.biig_json.training_samples, no_tf_transforms),
                                                 output_types=tuple([tf.string] + self.dst_type))
        #dataset = tf.data.Dataset.from_generator(lambda: self._data_generator(self.biig_json.training_samples),
                                                 #output_types=(tf.string, tf.string, tf.string), 
                                                #output_shapes =  (tf.TensorShape([]), tf.TensorShape([]), tf.TensorShape([])) )
        
        dataset = dataset.shard(self._num_gpus, self._gpu_id)
        #dataset = dataset.repeat()
        dataset = dataset.shuffle(self._batch_size * 3)
        
        if augment:        
            dataset = dataset.map(self._augment, num_parallel_calls=multiprocessing.cpu_count()//self._num_gpus)
            
        dataset = dataset.cache()
        dataset = dataset.repeat() # aquí o arriba?
        dataset = dataset.batch(self._batch_size, drop_remainder=False)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset
                     
    def eval_fn(self, samples=None, repeat=True):
        def resize_sample(image, label):
            return resize_image(image, self.dst_size, self.interpolator), resize_image(label, self.dst_size, _SITK_INTERPOLATOR_DICT.get("nearest"))
        samples = self.biig_json.validation_samples if samples is None else samples
        no_tf_transforms = [resize_sample,
                            lambda x,y: (sitk_to_np(x), sitk_to_np(y)),
                            lambda x,y: (x,treat_non_contiguous_labels(y)),
                            lambda x,y: (self.modality_normalization(x), y)]
        
        dataset = tf.data.Dataset.from_generator(lambda:self.preprocesing_transforms(samples, no_tf_transforms), output_types=tuple([tf.string] + self.dst_type))
        
        dataset = dataset.cache()
        if repeat:
            dataset = dataset.repeat()
        dataset = dataset.batch(self._batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset 
    
    def test_fn(self):
        pass
    

class SegmetationTestDataset(BiiGDataset):
    
    def _get_val_train_indices(self, length, fold, ratio=0.8):
        assert 0 < ratio <= 1, "Train/total data ratio must be in range (0.0, 1.0]"
        np.random.seed(self._seed)
        indices = np.arange(0, length, 1, dtype=np.int)
        np.random.shuffle(indices)
        if fold is not None:
            indices = deque(indices)
            indices.rotate(fold * int((1.0 - ratio) * length))
            indices = np.array(indices)
            train_indices = indices[:int(ratio * len(indices))]
            val_indices = indices[int(ratio * len(indices)):]
        else:
            train_indices = indices
            val_indices = []
        return train_indices, val_indices

    def _normalize_inputs(self, inputs):
        """Normalize inputs"""
        inputs = tf.expand_dims(tf.cast(inputs, tf.float32), -1)

        # Center around zero
        inputs = tf.divide(inputs, 127.5) - 1
        # Resize to match output size
        inputs = tf.image.resize(inputs, (388, 388))

        return tf.image.resize_with_crop_or_pad(inputs, 572, 572)

    def _normalize_labels(self, labels):
        """Normalize labels"""
        labels = tf.expand_dims(tf.cast(labels, tf.float32), -1)
        labels = tf.divide(labels, 255)

        # Resize to match output size
        labels = tf.image.resize(labels, (388, 388))
        labels = tf.image.resize_with_crop_or_pad(labels, 572, 572)

        cond = tf.less(labels, 0.5 * tf.ones(tf.shape(input=labels)))
        labels = tf.where(cond, tf.zeros(tf.shape(input=labels)), tf.ones(tf.shape(input=labels)))

        return tf.one_hot(tf.squeeze(tf.cast(labels, tf.int32)), 2)

    @tf.function
    def _preproc_samples(self, inputs, labels, augment=True):
        """Preprocess samples and perform random augmentations"""
        inputs = self._normalize_inputs(inputs)
        labels = self._normalize_labels(labels)

        if augment:
            # Horizontal flip
            h_flip = tf.random.uniform([]) > 0.5
            inputs = tf.cond(pred=h_flip, true_fn=lambda: tf.image.flip_left_right(inputs), false_fn=lambda: inputs)
            labels = tf.cond(pred=h_flip, true_fn=lambda: tf.image.flip_left_right(labels), false_fn=lambda: labels)

            # Vertical flip
            v_flip = tf.random.uniform([]) > 0.5
            inputs = tf.cond(pred=v_flip, true_fn=lambda: tf.image.flip_up_down(inputs), false_fn=lambda: inputs)
            labels = tf.cond(pred=v_flip, true_fn=lambda: tf.image.flip_up_down(labels), false_fn=lambda: labels)

            # Prepare for batched transforms
            inputs = tf.expand_dims(inputs, 0)
            labels = tf.expand_dims(labels, 0)

            # Random crop and resize
            left = tf.random.uniform([]) * 0.3
            right = 1 - tf.random.uniform([]) * 0.3
            top = tf.random.uniform([]) * 0.3
            bottom = 1 - tf.random.uniform([]) * 0.3

            inputs = tf.image.crop_and_resize(inputs, [[top, left, bottom, right]], [0], (572, 572))
            labels = tf.image.crop_and_resize(labels, [[top, left, bottom, right]], [0], (572, 572))

            # Gray value variations

            # Adjust brightness and keep values in range
            inputs = tf.image.random_brightness(inputs, max_delta=0.2)
            inputs = tf.clip_by_value(inputs, clip_value_min=-1, clip_value_max=1)

            inputs = tf.squeeze(inputs, 0)
            labels = tf.squeeze(labels, 0)

        # Bring back labels to network's output size and remove interpolation artifacts
        labels = tf.image.resize_with_crop_or_pad(labels, target_width=388, target_height=388)
        cond = tf.less(labels, 0.5 * tf.ones(tf.shape(input=labels)))
        labels = tf.where(cond, tf.zeros(tf.shape(input=labels)), tf.ones(tf.shape(input=labels)))

        return inputs, labels
    
    def train_fn(self, augment = True):
        def preproc(id_data, imgs, lbls):
            return self._preproc_samples(imgs, lbls)
        no_tf_transforms = [lambda x,y: (sitk_to_np(x), sitk_to_np(y))]
        dataset = tf.data.Dataset.from_generator(lambda:self.preprocesing_transforms(self.biig_json.training_samples, no_tf_transforms), 
                                                 output_types=tuple([tf.string] + self.dst_type),  
                                                 output_shapes=(tf.TensorShape([]), tf.TensorShape(self.dst_size), tf.TensorShape(self.dst_size))) 
        
        dataset = dataset.shard(self._num_gpus, self._gpu_id)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(self._batch_size * 3)
        dataset = dataset.map(preproc, num_parallel_calls=multiprocessing.cpu_count()//self._num_gpus)
        dataset = dataset.batch(self._batch_size, drop_remainder=False)
        dataset = dataset.prefetch(self._batch_size)

        return dataset
          
    def eval_fn(self, count):
        def preproc(id_data, imgs, lbls):
            return self._preproc_samples(imgs, lbls)
        no_tf_transforms = [lambda x,y: (sitk_to_np(x), sitk_to_np(y))]
        dataset = tf.data.Dataset.from_generator(lambda:self.preprocesing_transforms(self.biig_json.validation_samples, no_tf_transforms), 
                                                 output_types=tuple([tf.string] + self.dst_type),  
                                                 output_shapes=(tf.TensorShape([]), tf.TensorShape(self.dst_size), tf.TensorShape(self.dst_size))) 
        
        dataset = dataset.repeat(count=count)
        dataset = dataset.map(preproc, num_parallel_calls=multiprocessing.cpu_count())
        dataset = dataset.batch(self._batch_size, drop_remainder=False)
        dataset = dataset.prefetch(self._batch_size)
        return dataset
    
    def test_fn(self, count = 1):
        def preproc(id_data, imgs):
            return self._normalize_inputs(imgs)
            #return self._preproc_samples(imgs, lbls)
        no_tf_transforms = [lambda x,y: (sitk_to_np(x)) ]
        dataset = tf.data.Dataset.from_generator(lambda:self.preprocesing_transforms(self.biig_json.test_samples, no_tf_transforms), 
                                                 output_types=tuple([tf.string, tf.uint8]),
                                                 output_shapes=(tf.TensorShape([]), tf.TensorShape(self.dst_size))  ) 
        
        dataset = dataset.repeat(count=count)
        dataset = dataset.map(preproc)
        dataset = dataset.batch(self._batch_size, drop_remainder=False)
        dataset = dataset.prefetch(self._batch_size)
        return dataset
        
    

        
        
if __name__ == '__main__':
    ds = SegmentationLUNADataSet(json_path = "/lusair/data/data_LUNA.json",
                                 data_dir="/lung-data/",
                                 training_keys=["Philips_Brilliance16_unknown", "TOSHIBA_Aquilion_V2.02ER003"], 
                                 test_keys=["TOSHIBA_Aquilion_V2.04ER001"], 
                                 meta_validation_keys=["Philips_Brilliance 64_unknown", "GE MEDICAL SYSTEMS_LightSpeed VCT_07MW18.4", "SIEMENS_Sensation 64_syngo CT 2007S"], 
                                 interpolator = "bspline", 
                                 shuffle_before_split=False, 
                                 test_sample_fields=["image"], 
                                 dst_size=(128, 128, 64), 
                                 batch_size=2)
    """"
    ds_2 = SegmetationTestDataset(json_path = "/Unet_tf2/data/unet_testing.json",
                                   data_dir="/Unet_tf2/data/", 
                                   training_keys=["training"], 
                                   test_keys=["test"], 
                                   images_as_n_dim=2, 
                                   test_sample_fields=all_sample_fields, 
                                   dst_size=(512,512),
                                   batch_size = 1)
    """
                                   
    
    dataset = SegmentationLUNADataSet(data_dir="/lung-data/",
                                      json_path="/lung-data/data_LUNA.json",
                                      training_keys=['GE MEDICAL SYSTEMS_LightSpeed16_06MW03.5',
                                                     'GE MEDICAL SYSTEMS_LightSpeed16_07MW11.10', 
                                                     'GE MEDICAL SYSTEMS_LightSpeed16_LightSpeedverrel',
                                                     'GE MEDICAL SYSTEMS_LightSpeed16_LightSpeedApps401.8_H4.0M4',
                                                     'GE MEDICAL SYSTEMS_LightSpeed16_LightSpeedApps405I.2_H4.0M5',
                                                     'GE MEDICAL SYSTEMS_LightSpeed16_LightSpeedApps400.2_H16M3',
                                                     'GE MEDICAL SYSTEMS_LightSpeed Ultra_LightSpeedApps308I.2_H3.1M5',
                                                     'GE MEDICAL SYSTEMS_LightSpeed Ultra_LightSpeedApps305.3_H3.1M4',
                                                     'GE MEDICAL SYSTEMS_LightSpeed Ultra_LightSpeedApps303.1_H3M4',
                                                     'GE MEDICAL SYSTEMS_LightSpeed Ultra_LightSpeedApps305.4_H3.1M4',
                                                     'GE MEDICAL SYSTEMS_LightSpeed Ultra_LightSpeedApps304.3_H3.1M3',
                                                     'GE MEDICAL SYSTEMS_LightSpeed Ultra_unknown',
                                                     'GE MEDICAL SYSTEMS_LightSpeed QX/i_LightSpeedApps10.5_2.8.2I_H1.3M4',
                                                     'GE MEDICAL SYSTEMS_LightSpeed QX/i_unknown',
                                                     'GE MEDICAL SYSTEMS_LightSpeed Pro 16_LightSpeedverrel',
                                                     'GE MEDICAL SYSTEMS_LightSpeed Pro 16_06MW03.5',
                                                     'GE MEDICAL SYSTEMS_LightSpeed Pro 16_07MW11.10',
                                                     'GE MEDICAL SYSTEMS_LightSpeed VCT_06MW03.4',
                                                     'GE MEDICAL SYSTEMS_LightSpeed VCT_07MW18.4',
                                                     'GE MEDICAL SYSTEMS_LightSpeed VCT_unknown',
                                                     'GE MEDICAL SYSTEMS_LightSpeed Plus_LightSpeedApps2.4.2_H2.4M5',
                                                     'GE MEDICAL SYSTEMS_LightSpeed Power_LightSpeedApps2.5_hp.me',
                                                     'GE MEDICAL SYSTEMS_LightSpeed Power_LightSpeedverrel',
                                                     'SIEMENS_Sensation 16_VA70C',
                                                     'SIEMENS_Sensation 16_VB10B',
                                                     'SIEMENS_Sensation 16_syngo CT 2006G',
                                                     'SIEMENS_Sensation 16_VA60B',
                                                     'SIEMENS_Sensation 64_syngo CT 2005A',
                                                     'SIEMENS_Sensation 64_syngo CT 2006A',
                                                     'SIEMENS_Sensation 64_syngo CT 2007S',
                                                     'SIEMENS_Sensation 64_syngo CT 2005A0',
                                                     'SIEMENS_Definition_syngo CT 2007C',
                                                     'SIEMENS_Emotion 6_VA70A',
                                                     'Philips_Brilliance 16P_unknown',
                                                     'Philips_Brilliance 40_unknown',
                                                     'Philips_Brilliance 64_unknown',
                                                     'Philips_Brilliance16_unknown',
                                                     'TOSHIBA_Aquilion_V2.04ER001',
                                                     'TOSHIBA_Aquilion_V2.02ER003'], 
                                      test_keys=["TOSHIBA_Aquilion_V2.04ER001"], 
                                      meta_validation_keys=['Philips_Brilliance 16P_unknown',
                                                            'Philips_Brilliance 40_unknown',
                                                            'Philips_Brilliance 64_unknown',
                                                            'Philips_Brilliance16_unknown',
                                                            'TOSHIBA_Aquilion_V2.04ER001',
                                                            'TOSHIBA_Aquilion_V2.02ER003'],
                                      train_split = 1,
                                      interpolator = "bspline", 
                                      shuffle_before_split=False, 
                                      test_sample_fields=["image"], 
                                      dst_size=(128, 128, 64), 
                                      batch_size=4)
    import pandas as pd
    lista = []
    for sample in dataset.biig_json.training_samples:
        id_data = sample[0]
        image = sample[1]
        label = sample[2]
        print("Opening image "+image)
        image = sitk.ReadImage(image)
        image_np = sitk.GetArrayFromImage(image)
        mean = np.mean(image_np)
        median = np.median(image_np)
        minim = np.min(image_np)
        maxi = np.max(image_np)
        image_dict = {"file":sample[1],"id": id_data, "mean":mean, "median":median, "min":minim, "max":maxi, "unique":0}
        print(image_dict)
        print("Opening label "+label)
        label = sitk.ReadImage(label)
        label_np = sitk.GetArrayFromImage(label)
        mean_label = np.mean(label_np)
        median_label = np.median(label_np)
        minim_label = np.min(label_np)
        maxi_label = np.max(label_np)
        uni = np.unique(label_np)
        label_dict = {"file":sample[2],"id": id_data, "mean":mean_label, "median":median_label, "min":minim_label, "max":maxi_label, "unique":uni}
        print(label_dict)
        print("")
        lista.append(image_dict)
        lista.append(label_dict)
    df = pd.DataFrame(lista)
    df.to_csv("/lusair/data/stats_LUNA.csv")
        
    
    
    #fds = ds.train_fn(True)
    print("Iterating over samples")
    for iteration, (id_data,images, labels) in enumerate(ds.train_fn(augment=True)):
        break
        
    #fds_eval = ds.eval_fn()
    print("Iterating over samples")
    for iteration_eval, (id_data_eval,images_eval, labels_eval) in enumerate(ds.eval_fn()):
        break
    
    """
 
        
   
    print("Iterating over samples training")
    lst = []
    lst_2 = []
    for iteration, (images, labels) in enumerate(ds_2.train_fn(augment=True)):
        lst.append(images)
        lst_2.append(labels)
        if iteration > 30:
            break
        #print(iteration, images[0][0][0], labels[0][0][0])
        
    
    print("Iterating over samples validation")
    for iteration, (images_val, labels_val) in enumerate(ds_2.eval_fn(count=1)):
        break
    print("ITER_VAL", iteration, np.unique(labels_val))
        
    
    print("Iterating over samples testing")
    for iteration_test, (images_test) in enumerate(ds_2.test_fn(count=1)):
        break
        
    print("ITER_TEST", iteration_test, np.unique(images_test))
    
    """
    
    
