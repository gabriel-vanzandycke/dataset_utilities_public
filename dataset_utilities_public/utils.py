from dataclasses import dataclass
import io
import os
import random
import subprocess

import numpy as np
import cv2
import m3u8
import imageio

from mlworkflow import Dataset, AugmentedDataset, SideRunner
from mlworkflow.datasets import batchify

from .calib import Point3D

class RobustBatchesDataset(Dataset):
    def __init__(self, parent):
        self.parent = parent
    def yield_keys(self):
        yield from self.parent.yield_keys()
    def query_item(self, key):
        return self.parent.query_item(key)
    def chunkify(self, keys, chunk_size):
        d = {}
        for k, v in ((k,v) for k,v in ((k, self.query_item(k)) for k in keys) if v is not None):
            d[k] = v
            if len(d) == chunk_size:  # yield complete sublist and create a new list
                yield d
                d = {}
    @staticmethod
    def chunkify2(kv_gen, chunk_size):
        d = [None]*chunk_size
        for i, (k, v) in enumerate(kv_gen):
            if v is None:
                continue
            d[i] = (k, v)
            if i % chunk_size:
                yield d
                d = [None]*chunk_size
    def batches(self, keys, batch_size, wrapper=np.array, drop_incomplete=False):
        for dict_chunk in self.chunkify(keys, chunk_size=batch_size):
            yield list(dict_chunk.keys()), batchify(list(dict_chunk.values()))


def gamma_correction(image, gammas=np.array([1.0, 1.0, 1.0])):
    image = image.astype(np.float32)
    image = image ** (1/gammas)
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

class IncompatibleCropException(ValueError):
    pass


class RandomCropper():
    MIN_MARGIN = 2
    def __init__(self, output_shape, def_min, def_max, margin, max_angle, random_position=True):
        """margin - margin in cm arount the target center
        """
        self.output_shape = output_shape
        self.def_min = def_min
        self.def_max = def_max
        self.margin = margin
        self.max_angle = max_angle
        self.random_position = random_position
    @property
    def config(self):
        return {
            "output_shape": self.output_shape, "def_min": self.def_min, "def_max": self.def_max, "margin": self.margin,
            "max_angle": self.max_angle, "random_position": self.random_position
        }
    def __call__(self, calib, keypoints, seed, def_min=None, def_max=None, margin=None):
        random_state = np.random.get_state()
        np.random.seed(seed)

        def_min = def_min or self.def_min
        def_max = def_max or self.def_max
        margin = margin or self.margin

        if def_min > def_max:
            if margin < self.MIN_MARGIN:
                raise IncompatibleCropException("Impossible to decrease margin furthermore")
            return self(calib, keypoints, seed=seed+1, margin=margin//2)

        new_calib = calib

        wanted_definition = np.random.uniform(def_min, def_max)
        actual_definition = new_calib.compute_length2D(100, Point3D(np.mean(keypoints, axis=1)))
        scale = actual_definition/wanted_definition
        tmp_width, tmp_height = tuple(int(x*scale) for x in self.output_shape)

        # If wanted definition makes the output image bigger than input image, try with a bigger definition
        if tmp_width >= new_calib.width or tmp_height >= new_calib.height:
            return self(calib, keypoints, seed=seed+1, def_min=wanted_definition+10, margin=margin)

        margin_2D = new_calib.compute_length2D(margin, Point3D(np.mean(keypoints, axis=1)))
        keypoints_2D = new_calib.project_3D_to_2D(keypoints)

        if self.random_position:
            x_offset_min = max(0, min(int(np.max(keypoints_2D.x)+margin_2D), new_calib.width) - tmp_width)
            x_offset_max = min(new_calib.width - tmp_width, max(0, int(np.min(keypoints_2D.x)-margin_2D)))
            y_offset_min = max(0, min(int(np.max(keypoints_2D.y)+margin_2D), new_calib.height) - tmp_height)
            y_offset_max = min(new_calib.height - tmp_height, max(0, int(np.min(keypoints_2D.y)-margin_2D)))

            # If margin margin doesn't allow to use that definition, try with a lower margin
            if x_offset_max <= x_offset_min or y_offset_max <= y_offset_min:
                if margin < self.MIN_MARGIN :
                    raise IncompatibleCropException("Impossible to decrease margin furthermore")
                return self(calib, keypoints, seed=seed+1, def_min=def_min, margin=margin//2)

            x_offset = np.random.randint(x_offset_min, x_offset_max)
            y_offset = np.random.randint(y_offset_min, y_offset_max)
        else:
            x_offset = int(np.mean(keypoints_2D.x)-tmp_width/2)
            y_offset = int(np.mean(keypoints_2D.y)-tmp_height/2)
            if x_offset < 0 or y_offset < 0 or x_offset+tmp_width >= new_calib.width or y_offset+tmp_height >= new_calib.height:
                return self(calib, keypoints, seed=seed+1, def_min=wanted_definition+10, margin=margin)

        x_slice = slice(x_offset, x_offset+tmp_width, None)
        y_slice = slice(y_offset, y_offset+tmp_height, None)

        angle = self.max_angle*(2*np.random.beta(2, 2)-1)
        new_calib = calib.rotate(angle)

        np.random.set_state(random_state)

        return angle, x_slice, y_slice


class VideoReaderDataset(Dataset):
    cap = None
    def __init__(self, filename, scale_factor=None, output_shape=None):
        assert not scale_factor or not output_shape, "You cannot provide both 'scale_factor' and 'output_shape' arguments."
        self.cap = cv2.VideoCapture(filename)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        shape = tuple([int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))])
        if scale_factor:
            shape = tuple(x*scale_factor for x in shape)
        elif output_shape:
            shape = output_shape
        self.shape = tuple(x-x%2 for x in shape) # make sure shape is even
    def __del__(self):
        if self.cap is not None:
            self.cap.release()
    def yield_keys(self):
        yield from range(self.frame_count)
    def query_item(self, i):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        _, frame = self.cap.read()
        if frame is None:
            return None
        frame = cv2.resize(frame, self.shape)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame


class M3u8PlaylistDataset(Dataset):
    def __init__(self, filename):
        self.playlist = m3u8.load(filename)
    def yield_keys(self):
        yield from self.playlist.segments
    def query_item(self, key):
        return key.uri

class VideoFileNameToDatasetReaderTransform():
    def __init__(self, **kwargs):
        self.kwargs = kwargs
    def __call__(self, key, filename):
        return VideoReaderDataset(filename, **self.kwargs)

class VideoFromPlaylistDataset(AugmentedDataset):
    def root_key(self, key):
        return key[0]
    def augment(self, root_key, dataset):
        for key in dataset.yield_keys():
            item = dataset.query_item(key)
            if item is not None:
                yield (root_key, root_key.uri, key), item

class DatasetSamplerDataset(Dataset):
    def __init__(self, dataset, count):
        self.dataset = dataset
        self.keys = random.sample(list(dataset.keys.all()), count)
    def yield_keys(self):
        for key in self.keys:
            yield key
    def query_item(self, key):
        return self.dataset.query_item(key)



def concatenate_chunks(output_filename, *chunk_urls):
    side_runner = SideRunner(10)
    for chunk_url in chunk_urls:
        side_runner.run_async(subprocess.run, ["wget", chunk_url])
    side_runner.collect_runs()

    command = [
        'ffmpeg',
        '-y',
        '-protocol_whitelist "concat,file,http,https,tcp,tls"',
        '-i "concat:{}"'.format("|".join([url[url.rfind("/")+1:] for url in chunk_urls])),
        '-c:a copy',
        '-c:v copy',
        '-movflags faststart',
        output_filename
    ]
    os.system(" ".join(command))
    #subprocess.run(command) # For obscure reason, subprocess doesn't work here

@dataclass
class BoundingBox:
    x: int
    y: int
    w: int
    h: int
    @property
    def x_slice(self):
        return slice(self.x, self.x+self.w, None)
    @property
    def y_slice(self):
        return slice(self.y, self.y+self.h, None)

    def increase_box(self, max_width, max_height, aspect_ratio=None, margin=0):
        """ Adapt the bounding-box s.t. it
                - is increased by `margin` on all directions
                - lies within the source image of size `max_width`x`max_height`
                - has the aspect ratio given by `aspect_ratio` (if not None)
                - contains the original bounding-box (box is increased if necessary, up to source image limits)
            Arguments:
                max_width (int)      - width of input image
                max_height (int)     - height of input image
                aspect_ratio (float) - output aspect-ratio
                margin (int)         - margin in pixels to be added on 4 sides
            Returns:
                x_slice (slice) - the horizontal slice
                y_slice (slice) - the vertical slice
        """
        top   = max(0, int(self.y-margin))
        bot   = min(max_height, int(self.y+self.h+margin))
        left  = max(0, int(self.x-margin))
        right = min(max_width, int(self.x+self.w+margin))

        if aspect_ratio is None:
            return slice(left, right, None), slice(top, bot, None)

        w = right - left
        h = bot - top
        if w/h > aspect_ratio: # box is wider
            h = int(w/aspect_ratio)
            if h > max_height: # box is too wide
                h = max_height
                w = int(max_height*aspect_ratio)
                left = max_width//2 - w//2
                return slice(left, w, None), slice(0, h, None)
            cy = (bot+top)//2
            if cy + h//2 > max_height: # box hits the top
                return slice(left, right, None), slice(0, h, None)
            if cy - h//2 < 0: # box hits the bot
                return slice(left, right, None), slice(max_height-h, max_height, None)
            return slice(left, right, None), slice(cy-h//2, cy-h//2+h, None)

        if w/h < aspect_ratio: # box is taller
            w = int(h*aspect_ratio)
            if w > max_width: # box is too tall
                w = max_width
                h = int(max_width/aspect_ratio)
                top = max_height//2 - h//2
                return slice(0, w, None), slice(top, top+h, None)
            cx = (left+right)//2
            if cx + w//2 > max_width: # box hits the right
                return slice(max_width-w, max_width, None), slice(top, bot, None)
            if cx - w//2 < 0: # box hits the left
                return slice(0, w, None), slice(top, bot, None)
            return slice(cx-w//2, cx-w//2+w, None), slice(top, bot, None)

        # else: good aspect_ratio
        return slice(left, right, None), slice(top, bot, None)

class VideoMaker():
    format_map = {
        ".mp4": 'MP4V',
        ".avi": 'XVID'
    }
    writer = None
    def __init__(self, filename="output.mp4", framerate=15):
        self.filename = filename
        self.framerate = framerate
        self.fourcc = cv2.VideoWriter_fourcc(*self.format_map[filename[-4:]])
    def __enter__(self):
        return self
    def __call__(self, image):
        if self.writer is None:
            shape = (image.shape[1], image.shape[0])
            self.writer = cv2.VideoWriter(self.filename, self.fourcc, self.framerate, shape)
        self.writer.write(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer:
            self.writer.release()
            self.writer = None
            print("{} successfully written".format(self.filename))
    def __del__(self):
        if self.writer:
            self.writer.release()
            self.writer = None
            print("{} successfully written".format(self.filename))


# Image is 2D numpy array, q is quality 0-100
def jpegBlur(im, q):
    buf = io.BytesIO()
    imageio.imwrite(buf,im,format='jpg',quality=q)
    s = buf.getbuffer()
    return imageio.imread(s,format='jpg')


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

def setdefaultattr(obj, name, value):
    if not hasattr(obj, name):
        setattr(obj, name, value)
    return getattr(obj, name)
