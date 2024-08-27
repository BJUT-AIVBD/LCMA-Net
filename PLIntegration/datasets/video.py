#!usr/bin/env python
# -*- coding:utf-8 _*-
import warnings
from abc import ABCMeta, abstractmethod
from glob import glob
from typing import Union

# import mmcv
import numpy as np
import torch
import torchvision
from PIL import Image
# from mmaction.datasets.pipelines import Compose
# from mmcv import FileClient
import copy as cp
import os.path as osp

from torch.utils.data import Dataset
from torchvision.transforms import transforms


# class DecordDecode:
#     """Using decord to decode the video.
#
#     Decord: https://github.com/dmlc/decord
#
#     Required keys are "video_reader", "filename" and "frame_inds",
#     added or modified keys are "imgs" and "original_shape".
#
#     Args:
#         mode (str): Decoding mode. Options are 'accurate' and 'efficient'.
#             If set to 'accurate', it will decode videos into accurate frames.
#             If set to 'efficient', it will adopt fast seeking but only return
#             key frames, which may be duplicated and inaccurate, and more
#             suitable for large scene-based video datasets. Default: 'accurate'.
#     """
#
#     def __init__(self, mode='accurate'):
#         self.mode = mode
#         assert mode in ['accurate', 'efficient']
#
#     def __call__(self, results):
#         """Perform the Decord decoding.
#
#         Args:
#             results (dict): The resulting dict to be modified and passed
#                 to the next transform in pipeline.
#         """
#         container = results['video_reader']
#
#         if results['frame_inds'].ndim != 1:
#             results['frame_inds'] = np.squeeze(results['frame_inds'])
#
#         frame_inds = results['frame_inds']
#
#         if self.mode == 'accurate':
#             imgs = container.get_batch(frame_inds).asnumpy()
#             imgs = list(imgs)
#         elif self.mode == 'efficient':
#             # This mode is faster, however it always returns I-FRAME
#             container.seek(0)
#             imgs = list()
#             for idx in frame_inds:
#                 container.seek(idx)
#                 frame = container.next()
#                 imgs.append(frame.asnumpy())
#
#         results['video_reader'] = None
#         del container
#
#         results['imgs'] = imgs
#         results['original_shape'] = imgs[0].shape[:2]
#         results['img_shape'] = imgs[0].shape[:2]
#
#         return results
#
#     def __repr__(self):
#         repr_str = f'{self.__class__.__name__}(mode={self.mode})'
#         return repr_str
#
#
# class RawFrameDecode:
#     """Load and decode frames with given indices.
#
#     Required keys are "frame_dir", "filename_tmpl" and "frame_inds",
#     added or modified keys are "imgs", "img_shape" and "original_shape".
#
#     Args:
#         io_backend (str): IO backend where frames are stored. Default: 'disk'.
#         decoding_backend (str): Backend used for image decoding.
#             Default: 'cv2'.
#         kwargs (dict, optional): Arguments for FileClient.
#     """
#
#     def __init__(self, io_backend='disk', decoding_backend='cv2', **kwargs):
#         self.io_backend = io_backend
#         self.decoding_backend = decoding_backend
#         self.kwargs = kwargs
#         self.file_client = None
#
#     def __call__(self, results):
#         """Perform the ``RawFrameDecode`` to pick frames given indices.
#
#         Args:
#             results (dict): The resulting dict to be modified and passed
#                 to the next transform in pipeline.
#         """
#         mmcv.use_backend(self.decoding_backend)
#
#         directory = results['frame_dir']
#         filename_tmpl = results['filename_tmpl']
#         modality = results['modality']
#
#         if self.file_client is None:
#             self.file_client = FileClient(self.io_backend, **self.kwargs)
#
#         imgs = list()
#
#         if results['frame_inds'].ndim != 1:
#             results['frame_inds'] = np.squeeze(results['frame_inds'])
#
#         offset = results.get('offset', 0)
#
#         cache = {}
#         for i, frame_idx in enumerate(results['frame_inds']):
#             # Avoid loading duplicated frames
#             if frame_idx in cache:
#                 if modality == 'RGB':
#                     imgs.append(cp.deepcopy(imgs[cache[frame_idx]]))
#                 else:
#                     imgs.append(cp.deepcopy(imgs[2 * cache[frame_idx]]))
#                     imgs.append(cp.deepcopy(imgs[2 * cache[frame_idx] + 1]))
#                 continue
#             else:
#                 cache[frame_idx] = i
#
#             frame_idx += offset
#             if modality == 'RGB':
#                 filepath = osp.join(directory, filename_tmpl.format(frame_idx))
#                 img_bytes = self.file_client.get(filepath)
#                 # Get frame with channel order RGB directly.
#                 cur_frame = mmcv.imfrombytes(img_bytes, channel_order='rgb')
#                 imgs.append(cur_frame)
#             elif modality == 'Flow':
#                 x_filepath = osp.join(directory,
#                                       filename_tmpl.format('x', frame_idx))
#                 y_filepath = osp.join(directory,
#                                       filename_tmpl.format('y', frame_idx))
#                 x_img_bytes = self.file_client.get(x_filepath)
#                 x_frame = mmcv.imfrombytes(x_img_bytes, flag='grayscale')
#                 y_img_bytes = self.file_client.get(y_filepath)
#                 y_frame = mmcv.imfrombytes(y_img_bytes, flag='grayscale')
#                 imgs.extend([x_frame, y_frame])
#             else:
#                 raise NotImplementedError
#
#         results['imgs'] = imgs
#         results['original_shape'] = imgs[0].shape[:2]
#         results['img_shape'] = imgs[0].shape[:2]
#
#         # we resize the gt_bboxes and proposals to their real scale
#         if 'gt_bboxes' in results:
#             h, w = results['img_shape']
#             scale_factor = np.array([w, h, w, h])
#             gt_bboxes = results['gt_bboxes']
#             gt_bboxes = (gt_bboxes * scale_factor).astype(np.float32)
#             results['gt_bboxes'] = gt_bboxes
#             if 'proposals' in results and results['proposals'] is not None:
#                 proposals = results['proposals']
#                 proposals = (proposals * scale_factor).astype(np.float32)
#                 results['proposals'] = proposals
#
#         return results
#
#     def __repr__(self):
#         repr_str = (f'{self.__class__.__name__}('
#                     f'io_backend={self.io_backend}, '
#                     f'decoding_backend={self.decoding_backend})')
#         return repr_str
#
#
# class SampleFrames:
#     """Sample frames from the video.
#
#     Required keys are "total_frames", "start_index" , added or modified keys
#     are "frame_inds", "frame_interval" and "num_clips".
#
#     Args:
#         clip_len (int): Frames of each sampled output clip.
#         frame_interval (int): Temporal interval of adjacent sampled frames.
#             Default: 1.
#         num_clips (int): Number of clips to be sampled. Default: 1.
#         temporal_jitter (bool): Whether to apply temporal jittering.
#             Default: False.
#         twice_sample (bool): Whether to use twice sample when testing.
#             If set to True, it will sample frames with and without fixed shift,
#             which is commonly used for testing in TSM model. Default: False.
#         out_of_bound_opt (str): The way to deal with out of bounds frame
#             indexes. Available options are 'loop', 'repeat_last'.
#             Default: 'loop'.
#         test_mode (bool): Store True when building test or validation dataset.
#             Default: False.
#         start_index (None): This argument is deprecated and moved to dataset
#             class (``BaseDataset``, ``VideoDatset``, ``RawframeDataset``, etc),
#             see this: https://github.com/open-mmlab/mmaction2/pull/89.
#         keep_tail_frames (bool): Whether to keep tail frames when sampling.
#             Default: False.
#     """
#
#     def __init__(self,
#                  clip_len,
#                  frame_interval=1,
#                  num_clips=1,
#                  temporal_jitter=False,
#                  twice_sample=False,
#                  out_of_bound_opt='loop',
#                  test_mode=False,
#                  start_index=None,
#                  keep_tail_frames=False):
#
#         self.clip_len = clip_len
#         self.frame_interval = frame_interval
#         self.num_clips = num_clips
#         self.temporal_jitter = temporal_jitter
#         self.twice_sample = twice_sample
#         self.out_of_bound_opt = out_of_bound_opt
#         self.test_mode = test_mode
#         self.keep_tail_frames = keep_tail_frames
#         assert self.out_of_bound_opt in ['loop', 'repeat_last']
#
#         if start_index is not None:
#             warnings.warn('No longer support "start_index" in "SampleFrames", '
#                           'it should be set in dataset class, see this pr: '
#                           'https://github.com/open-mmlab/mmaction2/pull/89')
#
#     def _get_train_clips(self, num_frames):
#         """Get clip offsets in train mode.
#
#         It will calculate the average interval for selected frames,
#         and randomly shift them within offsets between [0, avg_interval].
#         If the total number of frames is smaller than clips num or origin
#         frames length, it will return all zero indices.
#
#         Args:
#             num_frames (int): Total number of frame in the video.
#
#         Returns:
#             np.ndarray: Sampled frame indices in train mode.
#         """
#         ori_clip_len = self.clip_len * self.frame_interval
#
#         if self.keep_tail_frames:
#             avg_interval = (num_frames - ori_clip_len + 1) / float(
#                 self.num_clips)
#             if num_frames > ori_clip_len - 1:
#                 base_offsets = np.arange(self.num_clips) * avg_interval
#                 clip_offsets = (base_offsets + np.random.uniform(
#                     0, avg_interval, self.num_clips)).astype(np.int)
#             else:
#                 clip_offsets = np.zeros((self.num_clips,), dtype=np.int)
#         else:
#             avg_interval = (num_frames - ori_clip_len + 1) // self.num_clips
#
#             if avg_interval > 0:
#                 base_offsets = np.arange(self.num_clips) * avg_interval
#                 clip_offsets = base_offsets + np.random.randint(
#                     avg_interval, size=self.num_clips)
#             elif num_frames > max(self.num_clips, ori_clip_len):
#                 clip_offsets = np.sort(
#                     np.random.randint(
#                         num_frames - ori_clip_len + 1, size=self.num_clips))
#             elif avg_interval == 0:
#                 ratio = (num_frames - ori_clip_len + 1.0) / self.num_clips
#                 clip_offsets = np.around(np.arange(self.num_clips) * ratio)
#             else:
#                 clip_offsets = np.zeros((self.num_clips,), dtype=np.int)
#
#         return clip_offsets
#
#     def _get_test_clips(self, num_frames):
#         """Get clip offsets in test mode.
#
#         Calculate the average interval for selected frames, and shift them
#         fixedly by avg_interval/2. If set twice_sample True, it will sample
#         frames together without fixed shift. If the total number of frames is
#         not enough, it will return all zero indices.
#
#         Args:
#             num_frames (int): Total number of frame in the video.
#
#         Returns:
#             np.ndarray: Sampled frame indices in test mode.
#         """
#         ori_clip_len = self.clip_len * self.frame_interval
#         avg_interval = (num_frames - ori_clip_len + 1) / float(self.num_clips)
#         if num_frames > ori_clip_len - 1:
#             base_offsets = np.arange(self.num_clips) * avg_interval
#             clip_offsets = (base_offsets + avg_interval / 2.0).astype(np.int)
#             if self.twice_sample:
#                 clip_offsets = np.concatenate([clip_offsets, base_offsets])
#         else:
#             clip_offsets = np.zeros((self.num_clips,), dtype=np.int)
#         return clip_offsets
#
#     def _sample_clips(self, num_frames):
#         """Choose clip offsets for the video in a given mode.
#
#         Args:
#             num_frames (int): Total number of frame in the video.
#
#         Returns:
#             np.ndarray: Sampled frame indices.
#         """
#         if self.test_mode:
#             clip_offsets = self._get_test_clips(num_frames)
#         else:
#             clip_offsets = self._get_train_clips(num_frames)
#
#         return clip_offsets
#
#     def __call__(self, results):
#         """Perform the SampleFrames loading.
#
#         Args:
#             results (dict): The resulting dict to be modified and passed
#                 to the next transform in pipeline.
#         """
#         total_frames = results['total_frames']
#
#         clip_offsets = self._sample_clips(total_frames)
#         frame_inds = clip_offsets[:, None] + np.arange(
#             self.clip_len)[None, :] * self.frame_interval
#         frame_inds = np.concatenate(frame_inds)
#
#         if self.temporal_jitter:
#             perframe_offsets = np.random.randint(
#                 self.frame_interval, size=len(frame_inds))
#             frame_inds += perframe_offsets
#
#         frame_inds = frame_inds.reshape((-1, self.clip_len))
#         if self.out_of_bound_opt == 'loop':
#             frame_inds = np.mod(frame_inds, total_frames)
#         elif self.out_of_bound_opt == 'repeat_last':
#             safe_inds = frame_inds < total_frames
#             unsafe_inds = 1 - safe_inds
#             last_ind = np.max(safe_inds * frame_inds, axis=1)
#             new_inds = (safe_inds * frame_inds + (unsafe_inds.T * last_ind).T)
#             frame_inds = new_inds
#         else:
#             raise ValueError('Illegal out_of_bound option.')
#
#         start_index = results['start_index']
#         frame_inds = np.concatenate(frame_inds) + start_index
#         results['frame_inds'] = frame_inds.astype(np.int)
#         results['clip_len'] = self.clip_len
#         results['frame_interval'] = self.frame_interval
#         results['num_clips'] = self.num_clips
#         return results
#
#     def __repr__(self):
#         repr_str = (f'{self.__class__.__name__}('
#                     f'clip_len={self.clip_len}, '
#                     f'frame_interval={self.frame_interval}, '
#                     f'num_clips={self.num_clips}, '
#                     f'temporal_jitter={self.temporal_jitter}, '
#                     f'twice_sample={self.twice_sample}, '
#                     f'out_of_bound_opt={self.out_of_bound_opt}, '
#                     f'test_mode={self.test_mode})')
#         return repr_str
#
#
# class BaseDataset(Dataset, metaclass=ABCMeta):
#     """Base class for datasets.
#
#     All datasets to process video should subclass it.
#     All subclasses should overwrite:
#
#     - Methods:`load_annotations`, supporting to load information from an
#     annotation file.
#     - Methods:`prepare_train_frames`, providing train data.
#     - Methods:`prepare_test_frames`, providing test data.
#
#     Args:
#         ann_file (str): Path to the annotation file.
#         pipeline (list[dict | callable]): A sequence of data transforms.
#         data_prefix (str | None): Path to a directory where videos are held.
#             Default: None.
#         test_mode (bool): Store True when building test or validation dataset.
#             Default: False.
#         multi_class (bool): Determines whether the dataset is a multi-class
#             dataset. Default: False.
#         num_classes (int | None): Number of classes of the dataset, used in
#             multi-class datasets. Default: None.
#         start_index (int): Specify a start index for frames in consideration of
#             different filename format. However, when taking videos as input,
#             it should be set to 0, since frames loaded from videos count
#             from 0. Default: 1.
#         modality (str): Modality of data. Support 'RGB', 'Flow', 'Audio'.
#             Default: 'RGB'.
#         sample_by_class (bool): Sampling by class, should be set `True` when
#             performing inter-class data balancing. Only compatible with
#             `multi_class == False`. Only applies for training. Default: False.
#         power (float): We support sampling data with the probability
#             proportional to the power of its label frequency (freq ^ power)
#             when sampling data. `power == 1` indicates uniformly sampling all
#             data; `power == 0` indicates uniformly sampling all classes.
#             Default: 0.
#         dynamic_length (bool): If the dataset length is dynamic (used by
#             ClassSpecificDistributedSampler). Default: False.
#     """
#
#     def __init__(self,
#                  ann_file,
#                  pipeline,
#                  data_prefix=None,
#                  test_mode=False,
#                  multi_class=False,
#                  num_classes=None,
#                  start_index=1,
#                  modality='RGB',
#                  sample_by_class=False,
#                  power=0,
#                  dynamic_length=False):
#         super().__init__()
#
#         self.ann_file = ann_file
#         self.data_prefix = osp.realpath(
#             data_prefix) if data_prefix is not None and osp.isdir(
#             data_prefix) else data_prefix
#         self.test_mode = test_mode
#         self.multi_class = multi_class
#         self.num_classes = num_classes
#         self.start_index = start_index
#         self.modality = modality
#         self.sample_by_class = sample_by_class
#         self.power = power
#         self.dynamic_length = dynamic_length
#
#         assert not (self.multi_class and self.sample_by_class)
#
#         self.pipeline = Compose(pipeline)
#         self.video_infos = self.load_annotations()
#         if self.sample_by_class:
#             self.video_infos_by_class = self.parse_by_class()
#
#             class_prob = []
#             for _, samples in self.video_infos_by_class.items():
#                 class_prob.append(len(samples) / len(self.video_infos))
#             class_prob = [x ** self.power for x in class_prob]
#
#             summ = sum(class_prob)
#             class_prob = [x / summ for x in class_prob]
#
#             self.class_prob = dict(zip(self.video_infos_by_class, class_prob))
#
#     @abstractmethod
#     def load_annotations(self):
#         """Load the annotation according to ann_file into video_infos."""
#
#     def load_json_annotations(self):
#         """Load json annotation file to get video information."""
#         video_infos = mmcv.load(self.ann_file)
#         num_videos = len(video_infos)
#         path_key = 'frame_dir' if 'frame_dir' in video_infos[0] else 'filename'
#         for i in range(num_videos):
#             path_value = video_infos[i][path_key]
#             if self.data_prefix is not None:
#                 path_value = osp.join(self.data_prefix, path_value)
#             video_infos[i][path_key] = path_value
#             if self.multi_class:
#                 assert self.num_classes is not None
#             else:
#                 assert len(video_infos[i]['label']) == 1
#                 video_infos[i]['label'] = video_infos[i]['label'][0]
#         return video_infos
#
#     def prepare_train_frames(self, idx):
#         """Prepare the frames for training given the index."""
#         results = cp.deepcopy(self.video_infos[idx])
#         results['modality'] = self.modality
#         results['start_index'] = self.start_index
#
#         # prepare tensor in getitem
#         # If HVU, type(results['label']) is dict
#         if self.multi_class and isinstance(results['label'], list):
#             onehot = torch.zeros(self.num_classes)
#             onehot[results['label']] = 1.
#             results['label'] = onehot
#
#         return self.pipeline(results)
#
#     def __len__(self):
#         """Get the size of the dataset."""
#         return len(self.video_infos)
#
#     def __getitem__(self, idx):
#         """Get the sample for either training or testing given index."""
#
#         return self.prepare_train_frames(idx)
#
#
# class VideoDataset(BaseDataset):
#     """Video dataset for action recognition.
#
#     The dataset loads raw videos and apply specified transforms to return a
#     dict containing the frame tensors and other information.
#
#     The ann_file is a text file with multiple lines, and each line indicates
#     a sample video with the filepath and label, which are split with a
#     whitespace. Example of a annotation file:
#
#     .. code-block:: txt
#
#         some/path/000.mp4 1
#         some/path/001.mp4 1
#         some/path/002.mp4 2
#         some/path/003.mp4 2
#         some/path/004.mp4 3
#         some/path/005.mp4 3
#
#
#     Args:
#         ann_file (str): Path to the annotation file.
#         pipeline (list[dict | callable]): A sequence of data transforms.
#         start_index (int): Specify a start index for frames in consideration of
#             different filename format. However, when taking videos as input,
#             it should be set to 0, since frames loaded from videos count
#             from 0. Default: 0.
#         **kwargs: Keyword arguments for ``BaseDataset``.
#     """
#
#     def __init__(self, ann_file, pipeline, start_index=0, **kwargs):
#         super().__init__(ann_file, pipeline, start_index=start_index, **kwargs)
#
#     def load_annotations(self):
#         """Load annotation file to get video information."""
#         if self.ann_file.endswith('.json'):
#             return self.load_json_annotations()
#
#         video_infos = []
#         with open(self.ann_file, 'r') as fin:
#             for line in fin:
#                 line_split = line.strip().split()
#                 if self.multi_class:
#                     assert self.num_classes is not None
#                     filename, label = line_split[0], line_split[1:]
#                     label = list(map(int, label))
#                 else:
#                     filename, label = line_split
#                     label = int(label)
#                 if self.data_prefix is not None:
#                     filename = osp.join(self.data_prefix, filename)
#                 video_infos.append(dict(filename=filename, label=label))
#         return video_infos


# region    # <--- old ---> #

def get_list_label(root: str, frame_per_clip=16):
    dirs = sorted(glob(osp.join(root, "*", "*")))
    l_videos = []
    d_label = {}
    idx_counter = 0
    for dir in dirs:
        frames = sorted(glob(osp.join(dir, "*")))
        num_frames = len(frames)
        frames = frames[0:num_frames:num_frames // frame_per_clip]
        frames = ["/".join(frames[i].split("/")[-3:]) for i in range(frame_per_clip)]
        l_videos.append(frames)
        if frames[0].split('/')[0] not in d_label:
            d_label[frames[0].split('/')[0]] = idx_counter
            idx_counter += 1
    return l_videos, d_label


def get_list_label_reid(root: str, frame_per_clip=16):
    frames = sorted(glob(osp.join(root, "*", "*")))
    d_videos = {}
    l_videos = []
    d_label = {}
    idx_counter = 0
    for frame in frames:
        if frame[:-10] not in d_videos:
            d_videos[frame[:-10]] = [frame[-27:]]
        else:
            d_videos[frame[:-10]].append(frame[-27:])
    for video in list(d_videos.keys()):
        frames = d_videos[video]
        num_frames = len(frames)
        if num_frames < 0.5 * frame_per_clip:
            continue
        if num_frames < frame_per_clip:
            frames.extend(frames[::-1])
            frames = frames[:frame_per_clip]
        elif num_frames != frame_per_clip:
            start = np.random.randint(0, num_frames - frame_per_clip)
            frames = frames[start:start + frame_per_clip]
        l_videos.append(frames)
        if video.split('/')[-2] not in d_label:
            d_label[video.split('/')[-2]] = idx_counter
            idx_counter += 1
    return l_videos, d_label


class MotionDataset(Dataset):
    def __init__(self, base_dir, transformer=None, is_train=True, **kwargs):
        '''
        self.list_IDs	: list of strings (each string: utt key)
        self.labels		: dictionary (key: utt key, value: label integer)
        return_label	: (boolean)
        '''
        self.labels = {}
        if is_train:
            self.list_IDs, self.labels = get_list_label(base_dir)
        else:
            self.list_IDs, self.labels = get_list_label_reid(base_dir)
        self.base_dir = base_dir
        self.transformer = transformer

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        IDs = self.list_IDs[index]
        try:
            frames = []
            for ID in IDs:
                X = Image.open(osp.join(self.base_dir, ID))
                if self.transformer is not None:
                    X = self.transformer(X)
                else:
                    X = torchvision.transforms.ToTensor()(X)
                frames.append(X)
            X = torch.stack(frames, 1) / 255
        except Exception:
            raise ValueError('%s' % ID)

        y = self.labels[IDs[0].split("/")[0]]
        return X, y

    # endregion # <--- old ---> #


class TrialTestOld(MotionDataset):
    def __init__(self, base_dir, transformer: Union[torchvision.transforms.Compose, None] = None,
                 **kwargs):
        super().__init__(base_dir, transformer=transformer, **kwargs)

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        return img, label, self.list_IDs[idx][0]
