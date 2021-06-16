'''Synergy Sequence Dataset
    The items are provided by the Synergy session, while the frames are
    collected from the Keemotion S3 bucket

'''
from dataclasses import dataclass
import os
from typing import NamedTuple

import cv2
import numpy as np

from dataset_utilities.utils import BoundingBox
from dataset_utilities.transforms import Transform
from mlworkflow import lazyproperty, AugmentedDataset

from PIL import Image


@dataclass
class NEWSBoundingBox(BoundingBox):
    keypoints: list
    playerId: str
    playerJerseyNumber: int
    playerAffiliation: str
    label: str

    def serialize(self):
        raise NotImplementedError()

class SequenceKey(NamedTuple):
    sequence_id: int

class Sequence():
    def __init__(self, db_item, dataset_folder):
        self.dataset_folder = dataset_folder
        self.attributes = list(db_item.keys())
        self.sequence_id = db_item['id']
        for attr_name, attr_value in db_item.items():
            setattr(self, attr_name, attr_value)
        self.annotations = []
        self.tracklets = {}
        for num_frame, frame in enumerate(self.frames):
            frame_ids = [x.get('playerId', 'Zero') for x in frame['boundingBoxes']]
            frame_annotation = []
            for idx in sorted(enumerate(frame_ids), key=lambda x: x[1]):
                box = frame['boundingBoxes'][idx[0]]
                newsbb = NEWSBoundingBox(int(box.get('x', 0)),
                                         int(box.get('y', 0)),
                                         int(box['width']),
                                         int(box['height']),
                                         box['keyPoints'],
                                         box.get('playerId', None),
                                         box.get('playerJerseyNumber', None),
                                         box.get('playerAffiliation', None),
                                         box.get('label', None))
                frame_annotation.append(newsbb)
                dframeBB = {'frameId': num_frame,
                            'NEWSBoundingBox': newsbb}
                if idx[1] in self.tracklets:
                    self.tracklets[idx[1]].append(dframeBB)
                else:
                    self.tracklets[idx[1]] = [dframeBB]
            self.annotations.append(frame_annotation)

    def load_image(self, img_name):
        filename = os.path.join(self.dataset_folder, img_name)
        image = cv2.imread(filename)
        if image is None:
            raise FileNotFoundError(filename)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def get_thumbnails(self, tracklet_id, output_shape=None, margin=0):
        images = self.images
        for dd in self.tracklets[tracklet_id]:
            image = images[dd['frameId']]
            box = dd['NEWSBoundingBox']
            image_height, image_width, _ = image.shape

            aspect_ratio = output_shape[1]/output_shape[2] if output_shape else None
            x_slice, y_slice = box.increase_box(image_width, image_height,
                                                aspect_ratio=aspect_ratio,
                                                margin=margin)

            keypoints = correct_coordinates(box, (image_width, image_height))
            keypoints = [
                {
                    'x': int(kp['x']-x_slice.start),  # adjust for thumbnails
                    'y': int(kp['y']-y_slice.start)
                } for kp in box.keypoints]

            box = NEWSBoundingBox(box.x-x_slice.start, box.y-y_slice.start,
                                  box.w, box.h, keypoints,
                                  box.playerId,
                                  box.playerJerseyNumber,
                                  box.playerAffiliation,
                                  box.label)

            new_image = image[y_slice, x_slice]
            yield Thumbnail(new_image, box)

    @lazyproperty
    def images(self):
        return [self.load_image(img_name) for img_name in self.files]

    @property
    def files(self):
        for index, _ in enumerate(self.frames):
            yield "{}_{}.png".format(self.sequence_id, index)

    @property
    def key(self):
        return SequenceKey(self.sequence_id)

    @property
    def db_item(self):
        return {k:getattr(self, k) for k in self.attributes}


class ProducedSequencesDataset():
    items_type = Sequence


class ThumbnailKey(NamedTuple):
    sequence_key: SequenceKey
    frame_index: int
    box_index: int

@dataclass
class Thumbnail:
    image: np.ndarray
    box: NEWSBoundingBox

    def draw(self):
        cv2.rectangle(self.image, (self.box.x, self.box.y), (self.box.x+self.box.w, self.box.y+self.box.h), (255,255,255), thickness=2)
        for kp in self.box.keypoints:
            cv2.circle(self.image, (kp['x'], kp['y']), 4, (128,128,128), thickness=2)


class KeypointCorrectionTransform(Transform):
    def __call__(self, sequence_key: SequenceKey, sequence: Sequence):
        img_filename = next(sequence.files)
        filename = os.path.join(sequence.dataset_folder, img_filename)
        width, height = Image.open(filename).size
        for ann in sequence.annotations:
            for box in ann:
                keypoints = correct_coordinates(box, (width, height))
                box.keypoints = keypoints
        return sequence


class ThumbnailsDataset(AugmentedDataset):
    def __init__(self, produced_sequences_dataset, output_shape=None, rescale=False, margin=0):
        """
            output_shape - width x height
        """
        super().__init__(produced_sequences_dataset)
        self.output_shape = output_shape
        self.rescale = rescale
        self.margin = margin

    def augment(self, sequence_key: SequenceKey, sequence: Sequence):
        for image_index, (image, boxes) in enumerate(zip(sequence.images, sequence.annotations)):
            for box_index, box in enumerate(boxes):
                image_height, image_width, _ = image.shape

                aspect_ratio = self.output_shape[1]/self.output_shape[2] if self.output_shape else None
                x_slice, y_slice = box.increase_box(image_width, image_height, aspect_ratio=aspect_ratio, margin=self.margin)

                keypoints = correct_coordinates(box, (image_width, image_height))
                keypoints = [
                    {
                        'x': int(kp['x']-x_slice.start),  # adjust for thumbnails
                        'y': int(kp['y']-y_slice.start)
                    } for kp in box.keypoints
                ]

                box = NEWSBoundingBox(box.x-x_slice.start, box.y-y_slice.start,
                                      box.w, box.h, keypoints,
                                      box.get('playerId', None),
                                      box.get('playerJerseyNumber', None),
                                      box.get('playerAffiliation', None),
                                      box.get('label', None))

                new_image = image[y_slice, x_slice]
                if self.rescale:
                    new_image = cv2.resize(new_image, self.output_shape)
                    # TODO: resize keypoints as well
                    raise NotImplementedError('box and keypoints must be resized as well')
                yield ThumbnailKey(sequence_key, image_index, box_index), Thumbnail(new_image, box)

class ImagesDataset(AugmentedDataset):
    pass


def correct_coordinates(box, image_shape):
    """ Handles missing coordinates in the annotation
    when point was clicked outside image border

    Args:
        box : NEWSBoundingBox object
        image_shape : tuple (image_width, image_height)

    Return:
        corrected keypoints
    """
    img_width, img_height = image_shape

    return [
            {
             'x': int(kp.get('x', 0 if box.x < img_width//2 else img_width-1)),
             'y': int(kp.get('y', 0 if box.y < img_height//2 else img_height-1))
            } for kp in box.keypoints
            ]
