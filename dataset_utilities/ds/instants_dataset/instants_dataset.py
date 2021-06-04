import os
import json
from typing import NamedTuple
from enum import IntFlag

import cv2
import imageio
import numpy as np

from mlworkflow import lazyproperty

from dataset_utilities.calib import Calib, Point3D
from dataset_utilities.court import Court, court_dim as COURT_DIM

class DownloadFlags(IntFlag):
    NONE = 0
    WITH_IMAGES = 1
    WITH_CALIB_FILE = 2
    WITH_FOREGROUND_MASK_FILE = 4
    WITH_HUMAN_SEGMENTATION_MASKS = 8
    ALL = -1

class InstantKey(NamedTuple):
    arena_label: str
    game_id: int
    timestamp: int

class Instant():
    def __init__(self, db_item, dataset_folder, download_flags):
        self.dataset_folder = dataset_folder
        self.download_flags = download_flags

        self.arena_label = db_item["arena_label"]
        self.num_cameras = db_item["num_cameras"]

        self.game_id = db_item["game_id"]
        self.league_id = db_item["league_id"]
        self.rule_type = db_item["rule_type"]
        self.sport = db_item["sport"]

        self.timestamp = db_item["timestamp"]
        self.offsets = db_item["offsets"]

        self.annotation_state = db_item["annotation_state"]
        self.annotator_id = db_item.get("annotator_id", None)
        self.annotation_ts = db_item.get("annotation_ts", None)
        self.annotation_duration = db_item.get("annotation_duration", None)
        self.annotation_game_state = db_item.get("annotation_game_state", "standard_game")
        self.annotated_human_masks = db_item.get("annotated_human_masks", False)

        self.format =  db_item["format"]

        annotation_map = {
            "player": PlayerAnnotation,
            "ball": BallAnnotation
        }
        self.annotations = [annotation_map[a['type']](a)for a in (db_item.get('annotations', []) or [])]

        self.has_fgmask = db_item['fgmask']
        self.image_source = db_item.get("image_source", "raw")
        self.court_dim = COURT_DIM[self.rule_type]

    def __str__(self):
        return "({}[{:5d}]@{})".format(self.arena_label, self.game_id, self.timestamp)

    def get_filekey(self, prefix, suffix):
        return os.path.join(self.arena_label, str(self.game_id), "{}{}{}".format(prefix, self.timestamp, suffix))

    @lazyproperty
    def calibs(self):
        return [self.load_calib(c) for c in range(self.num_cameras)]

    @lazyproperty
    def all_images(self):
        all_images = {}
        for c in range(self.num_cameras):
            for offset in self.offsets:
                all_images[(c,offset)] = self.load_image(c, offset)
        return all_images

    @property
    def images(self):
        return [img for (c, offset), img in self.all_images.items() if offset == 0]

    @lazyproperty
    def foregrounds(self):
        assert self.download_flags & DownloadFlags.WITH_FOREGROUND_MASK_FILE, "Provided flag doesn't contain 'foregrounds'. Recreate your dataset with appropriate DownloadFlags"
        try:
            return [self.load_mask(c) for c in range(self.num_cameras)]
        except FileNotFoundError:
            return []

    @lazyproperty
    def fg_detections(self):
        return [self.load_detections(c) for c in range(self.num_cameras)]

    @lazyproperty
    def human_masks(self):
        assert self.download_flags & DownloadFlags.WITH_HUMAN_SEGMENTATION_MASKS, "Provided flag doesn't contain 'human_masks'. Recreate your dataset with appropriate DownloadFlags"
        try:
            filenames = [os.path.join(self.dataset_folder, self.get_filekey("camcourt{}_".format(cam_idx+1), "_humans.png")) for cam_idx in range(self.num_cameras)]
            return [imageio.imread(filename) for filename in filenames] # imageio handles 16bits images while cv2 doesn't
        except FileNotFoundError:
            # If one human_masks file is missing for one camera, no human_masks will be available.
            # If file is missing because no human appears on that camera, you should upload an empty image to the bucket.
            return []

    def load_image(self, cam_idx, offset=0):
        filename = os.path.join(self.dataset_folder, self.get_filekey("camcourt{}_".format(cam_idx+1), "_{}.png".format(offset)))
        image = cv2.imread(filename)
        if image is None:
            raise FileNotFoundError(filename)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def load_calib(self, cam_idx):
        filename = os.path.join(self.dataset_folder, self.get_filekey("camcourt{}_".format(cam_idx+1), ".json"))
        return Calib.parse_DeepSport(json.load(open(filename, 'r'))['calibration'])

    def load_detections(self, cam_idx):
        filename = os.path.join(self.dataset_folder, self.get_filekey("camcourt{}_".format(cam_idx+1), ".json"))
        return [ForegroundDetection(det, cam_idx) for det in json.load(open(filename, 'r'))["players"]]

    def load_mask(self, cam_idx):
        filename = os.path.join(self.dataset_folder, self.get_filekey("camcourt{}_".format(cam_idx+1), "_fgd.png"))
        try:
            return imageio.imread(filename)
        except BaseException as e:
            raise FileNotFoundError from e

    def draw(self, i=None, draw_players=True, draw_ball=True, draw_lines=False):
        if i is None:
            w, h = self.court_dim
            image = np.ones((int(h), int(w), 3), np.uint8)*255
            R = np.identity(3)
            C = Point3D(w/2, h/2, -3000)
            m = w/0.01  # pixel_size (pixels/meters) on 1 centimeter sensor
            f = 0.009  # focal (meters)
            R = np.identity(3)
            calib = Calib(width=w, height=h, T=-R@C[:3], R=R, K=np.array([[f*m,  0, w/2], [0, f*m, h/2], [0,  0,  1]]))
        else:
            image = self.images[i].copy()
            calib = self.calibs[i]

        if draw_lines:
            Court(self.rule_type).draw_lines(image, calib)
        # if fg_detections is not None:
        #     calib = self.calibs[i] if i is not None else None
        #     detections = self.fg_detections[i][fg_detections] if i is not None else \
        #                  itertools.chain(*self.fg_detections)
        #     for det in detections:
        #         v = calib.project_3D_to_2D(det.feet).to_int_tuple()
        #         cv2.circle(image, v[0:2].flatten(), 7, [255, 0, 255], -1)

        for annotation in self.annotations:
            if annotation.type == "player" and draw_players:
                head = calib.project_3D_to_2D(annotation.head).to_int_tuple()
                hips = calib.project_3D_to_2D(annotation.hips).to_int_tuple()
                foot1 = calib.project_3D_to_2D(annotation.foot1).to_int_tuple()
                foot2 = calib.project_3D_to_2D(annotation.foot2).to_int_tuple()

                if any([kp[0] < 0 or kp[1] > image.shape[1] or kp[1] < 0 or kp[1] > image.shape[0] for kp in [head, hips]]):
                    continue

                # head tip
                length = 70 # cm
                headT3D = length * Point3D(np.cos(annotation.headAngle), np.sin(annotation.headAngle), 0)
                headT = calib.project_3D_to_2D(annotation.head+headT3D).to_int_tuple()

                color = [0, 0, 0]
                color[annotation.team-1] = 255

                if i is not None:
                    cv2.line(image, hips, foot1, color, 3)
                    cv2.line(image, hips, foot2, color, 3)
                    cv2.line(image, head, hips, color, 3)
                else:
                    cv2.circle(image, head, 5, color, -1)
                cv2.line(image, head, headT, color, 3)

            elif annotation.type == "ball" and draw_ball:
                center = tuple(int(x) for x in calib.project_3D_to_2D(annotation.center).to_list())
                color = [255, 255, 0]
                cv2.circle(image, center, 5, color, -1)
        return image

    @property
    def files(self):
        for i in range(0, int(self.num_cameras)):
            if self.download_flags & DownloadFlags.WITH_IMAGES:
                for offset in self.offsets:
                    yield self.get_filekey("camcourt{}_".format(i+1), "_{}.png".format(offset))
            if self.download_flags & DownloadFlags.WITH_CALIB_FILE:
                yield self.get_filekey("camcourt{}_".format(i+1), ".json")
            if self.download_flags & DownloadFlags.WITH_FOREGROUND_MASK_FILE:
                yield self.get_filekey("camcourt{}_".format(i+1), "_fgd.png")
            if self.download_flags & DownloadFlags.WITH_HUMAN_SEGMENTATION_MASKS:
                yield self.get_filekey("camcourt{}_".format(i+1), "_humans.png")

    @property
    def key(self):
        return InstantKey(self.arena_label, self.game_id, self.timestamp)

    @property
    def db_item(self):
        return {
            "format": self.format,
            "image_source": self.image_source,

            # arena relative infos
            "arena_label": self.arena_label,
            "num_cameras": self.num_cameras,

            # game relative infos
            "sport": self.sport,
            "game_id": self.game_id,
            "league_id": self.league_id,
            "rule_type": self.rule_type,

            # instant relative infos
            "timestamp": self.timestamp,
            "offsets": self.offsets,
            "fgmask": self.has_fgmask,

            # annotation relative infos
            "annotator_id": self.annotator_id,
            "annotation_state": self.annotation_state,
            "annotation_ts": self.annotation_ts,
            "annotation_duration": self.annotation_duration,
            "annotations": [a.to_annotation() for a in self.annotations],
            "annotation_game_state": self.annotation_game_state,
            "annotated_human_masks": len(self.human_masks) > 0
        }

    def to_dict(self):
        return {"db_item": self.db_item, "download_flags": self.download_flags, "dataset_folder": self.dataset_folder}

class InstantsDataset():
    items_type = Instant

class BallAnnotation():
    def __init__(self, annotation):
        self.type = "ball"
        self.center = Point3D(*annotation['center'])
        self.camera = annotation['image']
        self.visible = annotation.get('visible', True)

    def to_annotation(self):
        return {
            "type": self.type,
            "center": self.center.to_list(),
            "image": self.camera,
            "visible": self.visible
        }

class PlayerAnnotation():
    def __init__(self, annotation):
        self.type = "player"
        self.origin = "annotations"
        self.team = annotation['team']
        self.head = Point3D(*annotation['head'])
        self.hips = Point3D(*annotation['hips'])
        self.foot1 = Point3D(*annotation['foot1'])
        self.foot2 = Point3D(*annotation['foot2'])
        self.foot1_at_the_ground = str(annotation["foot1_at_the_ground"]).lower() == "true"
        self.foot2_at_the_ground = str(annotation["foot2_at_the_ground"]).lower() == "true"
        self.headAngle = annotation['headOrientation']
        self.camera = annotation['image']
        self.hipsAngle = annotation.get('hipsOrientation', self.headAngle)
        self.feet = (self.foot1 + self.foot2) / 2

    def to_annotation(self):
        return {
            "type": self.type,
            "team": self.team,
            "head": self.head.to_list(),
            "headOrientation": self.headAngle,
            "hips": self.hips.to_list(),
            "foot1": self.foot1.to_list(),
            "foot2": self.foot2.to_list(),
            "foot1_at_the_ground": self.foot1_at_the_ground,
            "foot2_at_the_ground": self.foot2_at_the_ground,
            "image": self.camera
        }

class ForegroundDetection():
    def __init__(self, detection, camera: int) -> None:
        self.origin = "foreground"
        self.feet = Point3D(*[detection["pos_feet"][0], detection["pos_feet"][1], 0])
        self.confidence = detection["level"]
        self.status = detection["status"]
        self.camera = camera

