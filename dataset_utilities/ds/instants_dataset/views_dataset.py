from typing import NamedTuple
from scipy.spatial import ConvexHull # pylint: disable=no-name-in-module
import cv2
import numpy as np
from mlworkflow import AugmentedDataset

from dataset_utilities.calib import Point3D, Point2D, Calib
from dataset_utilities.court import Court, court_dim
from dataset_utilities.utils import BoundingBox

from .instants_dataset import InstantsDataset, Instant, InstantKey, DownloadFlags

class ViewKey(NamedTuple):
    instant_key: InstantKey
    camera: int
    index: int

    @property
    def arena_label(self):
        return self.instant_key.arena_label

    @property
    def game_id(self):
        return self.instant_key.game_id

class View():
    def __init__(self, all_images, box, calib, annotations=None, **kwargs):
        self.image = all_images[0]
        self.all_images = all_images
        self.box = box
        self.calib = calib
        self.annotations = annotations
        for k, v in kwargs.items():
            setattr(self, k, v)

    def draw(self):
        image = self.image.copy()

        coord_2D_int = lambda x: tuple(int(v) for v in self.calib.project_3D_to_2D(x))

        for annotation in self.annotations:
            if annotation.type == "player":
                head = annotation.head
                hips = annotation.hips
                foot1 = annotation.foot1
                foot2 = annotation.foot2

                color = [0, 0, 0]
                color[annotation.team-1] = 255
                cv2.line(image, coord_2D_int(head), coord_2D_int(hips), color, 3)
                cv2.line(image, coord_2D_int(hips), coord_2D_int(foot1), color, 3)
                cv2.line(image, coord_2D_int(hips), coord_2D_int(foot2), color, 3)

            elif annotation.type == "ball":
                ball =  annotation.center
                color = [255, 255, 0]
                cv2.circle(image, coord_2D_int(ball), 5, color, -1)

        return image

class ViewCropper():
    def __init__(self, camera, index, box: BoundingBox, **kwargs):
        self.camera = camera
        self.index = index
        self.box = box
        self.data = kwargs

class ViewBuilder():
    margin = 0
    def __init__(self, margin=0):
        """
            margin in cm
        """
        self.margin = margin
    def compute_3D_margin(self, pt3D, calib: Calib, margindir):
        """
            margindir is a 2D vector indicating margin direction relative to the picture, but final length is given in cm by self.margin
        """
        poscam = calib.C
        pt2D = calib.project_3D_to_2D(pt3D)
        pt2Do = np.copy(pt2D) + margindir
        pt3Do = calib.project_2D_to_3D(Point2D(pt2Do), 0)
        # orthogonal projection of pt3D on this ray
        poscam_pt3D = np.subtract(pt3D, poscam)
        normed_poscam_pt3Do = np.subtract(pt3Do, poscam)
        normed_poscam_pt3Do = normed_poscam_pt3Do/np.linalg.norm(normed_poscam_pt3Do)
        poscam_pt3Do = np.dot(np.squeeze(np.asarray(poscam_pt3D)), np.squeeze(np.array(normed_poscam_pt3Do))) * normed_poscam_pt3Do
        # pt3D to pt3Do
        pt3D_pt3Do = -poscam_pt3D + poscam_pt3Do
        pt3D_pt3Do = pt3D_pt3Do/np.linalg.norm(pt3D_pt3Do)
        pt3D_pt3Do = self.margin * pt3D_pt3Do
        # new keypoint with margins
        return pt3D + Point3D(pt3D_pt3Do)

    def compute_box(self, point3D_list: list, calib: Calib):
        point_and_margin_2D = [(
                calib.project_3D_to_2D(point3D),
                calib.compute_length2D(self.margin, point3D)
            ) for point3D in point3D_list]
        left_idx  = np.argmin([d[0].x for d in point_and_margin_2D])
        right_idx = np.argmax([d[0].x for d in point_and_margin_2D])
        top_idx   = np.argmin([d[0].y for d in point_and_margin_2D])
        bot_idx   = np.argmax([d[0].y for d in point_and_margin_2D])

        left  = max(0,            point_and_margin_2D[left_idx][0].x - point_and_margin_2D[left_idx][1])
        right = min(calib.width,  point_and_margin_2D[right_idx][0].x + point_and_margin_2D[right_idx][1])
        top   = max(0,            point_and_margin_2D[top_idx][0].y - point_and_margin_2D[top_idx][1])
        bot   = min(calib.height, point_and_margin_2D[bot_idx][0].y + point_and_margin_2D[bot_idx][1])
        return BoundingBox(left, top, right-left, bot-top)

class BuildCameraViews(ViewBuilder):
    """ Builds a view for each camera (margin parameter is useless)
    """
    def __call__(self, instant_key: InstantKey, instant:Instant):
        for c in range(instant.num_cameras):
            yield ViewCropper(c, 0, BoundingBox(0, 0, instant.calibs[c].width, instant.calibs[c].height), court_dim=instant.court_dim)

class BuildCourtViews(ViewBuilder):
    def __init__(self, height=300):
        super().__init__()
        self.height = height
    """ Builds a view including all the court keypoints visible on each camera
        Note: keypoints are duplicated at 2m from the floor
    """
    def __call__(self, instant_key: InstantKey, instant:Instant):
        for c in range(instant.num_cameras):
            calib = instant.calibs[c]
            visible_edges = Court(instant.rule_type).visible_edges(calib)
            court_keypoints = []
            for p1, p2 in visible_edges:
                court_keypoints = court_keypoints + [p1, p1+Point3D(0,0,-self.height), p2, p2+Point3D(0,0,-self.height)]
            yield ViewCropper(c, 0, self.compute_box(court_keypoints, calib), rule_type=instant.rule_type)

class BuildPlayersViews(ViewBuilder):
    """ Builds a view around the players visible on each camera
        min_annotations: minimum required number of person to use that camera
    """
    def __init__(self, *, min_annotations=1, **kwargs):
        self.min_annotations = min_annotations
        super().__init__(**kwargs)

    def __call__(self, instant_key: InstantKey, instant: Instant):
        for c in range(instant.num_cameras):
            annotations = [a for a in instant.annotations if a.camera == c and a.type == "player" and a.team > 0]
            if len(annotations) < self.min_annotations:
                continue

            keypoints = []
            for a in annotations:
                keypoints += [a.head, a.hips, a.foot1, a.foot2]

            yield ViewCropper(c, 0, self.compute_box(keypoints, instant.calibs[c]))

class BuildThumbnailViews(ViewBuilder):
    BODY_HEIGHT = 180
    """ Builds a view around each person (players, referee)
    """
    def __init__(self, *, with_annotations=True, with_detections=False, with_random=0,
        with_occlusions=False, **kwargs):
        self.with_annotations = with_annotations
        self.with_detections = with_detections
        self.with_occlusions = with_occlusions
        self.with_random = 10 if isinstance(with_random, bool) and with_random else with_random
        self.threshold = 0.25
        super().__init__(**kwargs)

    def check_density_map(self, density_map, box, threshold):
        x_slice, y_slice = box
        if 0 in density_map[y_slice, x_slice].shape:
            return False
        if np.mean(density_map[y_slice, x_slice]) <= threshold:
            return True
        return False

    def sample_density_map(self, density_map):
        # avoid going too close to other detections
        dilation = cv2.dilate(density_map, np.ones((3,3)), iterations=10)
        indices_y, indices_x = np.where(dilation == 0)
        # random choic a position in the image
        i = np.random.randint(0, len(indices_x))
        return np.array([[indices_x[i], indices_y[i]]])

    @staticmethod
    def fill_density_map(density_map, box):
        x_slice, y_slice = box
        density_map[y_slice, x_slice] += 1

    def __call__(self, instant_key: InstantKey, instant:Instant):
        # Set random seed with timestamp
        random_state = np.random.get_state()
        np.random.seed(instant_key.timestamp & 0xFFFFFFFF)

        instant.density_maps = [np.zeros(img.shape[0:2], dtype=np.uint8) for img in instant.images]

        for c in range(instant.num_cameras):
            calib = instant.calibs[c]
            density_map = instant.density_maps[c]
            index = 0

            # From annotation
            for a in [a for a in instant.annotations if a.type == "player" and calib.projects_in(a.hips) and self.with_annotations]:
                keypoints = [a.head, a.hips, a.foot1, a.foot2]
                box = self.compute_box(keypoints, calib)
                #if self.check_density_map(instant.density_maps, box, c, 1+self.threshold) or self.with_occlusions:
                yield ViewCropper(c, index, box, origin='annotations', annotation=a, density_map=density_map)
                self.fill_density_map(density_map, box)
                index = index + 1

            if self.with_detections:
                # From keemotion foregrounddetector detections
                for detection in []:#[d for d in instant.fg_detections if d.camera == c and calib.projects_in(d.feet) and self.with_detections]:
                    keypoints = [ detection.feet, detection.feet + Point3D(0, 0, -self.BODY_HEIGHT) ]
                    box = self.compute_box(keypoints, calib)
                    if self.check_density_map(density_map, box, self.threshold):
                        yield ViewCropper(c, index, box, origin='detections', detection=detection, density_map=density_map)
                        self.fill_density_map(density_map, box)
                        index = index + 1


            # From random
            if self.with_random:
                raise NotImplementedError
                # TODO: use Calib.visible_edge() to replace the function "find_court_intersection_with_camera_border"
                court_keypoints_3D = []# find_court_intersection_with_camera_border(calib, instant.rule_type)
                court_keypoints_2D = np.array([calib.project_3D_to_2D(p).to_list() for p in court_keypoints_3D])
                convex_hull = ConvexHull(court_keypoints_2D)
                points = np.array([court_keypoints_2D[i,:] for i in convex_hull.vertices]).astype(np.int32)
                court = np.ones(instant.images[c].shape[0:2], dtype=np.uint8)
                density_map[cv2.fillPoly(court, [points], 0)==1] += 1

            for _ in range(self.with_random):
                feet = calib.project_2D_to_3D(Point2D(self.sample_density_map(density_map)), Z=0)
                keypoints = [feet, feet + Point3D(0, 0, -self.BODY_HEIGHT)]
                box = self.compute_box(keypoints, calib)
                if self.check_density_map(density_map, box, self.threshold):
                    yield ViewCropper(c, index, box, origin='random', density_map=density_map)
                    self.fill_density_map(density_map, box)
                    index = index + 1

        # Restore random seed
        np.random.set_state(random_state)

class BuildBallViews(ViewBuilder):
    def __call__(self, instant_key: InstantKey, instant: Instant):
        balls = [a for a in instant.annotations if a.type == "ball"]
        assert len(balls) == 1, "Too many balls annotated for {}.".format(instant_key)
        ball = balls[0]
        keypoints = [ball.center]
        c = int(ball.camera)
        yield ViewCropper(c, 0, self.compute_box(keypoints, instant.calibs[c]), annotation=ball)

class BuildHeadsViews(ViewBuilder):
    def __call__(self, instant_key: InstantKey, instant: Instant):
        for idx, player in enumerate([a for a in instant.annotations if a.type == "player"]):
            c = int(player.camera)
            keypoints = [player.head]
            yield ViewCropper(c, idx, self.compute_box(keypoints, instant.calibs[c]), annotation=player)

class ViewsDataset(AugmentedDataset):
    """ Dataset of views built according the given ViewBuilder, extracted from the given InstantDataset.
        Arguments:
            instants_dataset   - the InstantDataset from which views are built
            view_builder       - the ViewBuilder dictating what type of view is to be created
            output_shape       - view aspect ratio (or exact dimension if 'rescale' is given)
            rescale            - tells whether the view should be rescaled to output_shape size
            crop               - tells whether the original image should be cropped or a rectangle
                                 should be drawn instead (for debug purposes)
    """
    def __init__(self, instants_dataset: InstantsDataset, view_builder: ViewBuilder, output_shape=None, rescale=True, crop=True):
        super().__init__(instants_dataset)
        self.view_builder = view_builder
        self.output_shape = output_shape
        self.rescale = rescale
        self.crop = crop

    def _crop_view(self, view_cropper: ViewCropper, instant: Instant, **kwargs):
        c = view_cropper.camera
        input_width = instant.images[c].shape[1]
        input_height = instant.images[c].shape[0]
        ratio = lambda wh: wh[0]/wh[1]
        x_slice, y_slice = view_cropper.box.increase_box(input_width, input_height, ratio(self.output_shape) if self.output_shape else None)
        all_images = []

        if self.crop:
            for offset in instant.offsets:
                index = (c,offset)
                if index not in instant.all_images:
                    continue # that offset was not downloaded with the download flag of instants dataset
                image = instant.all_images[index][y_slice, x_slice]
                if self.rescale and self.output_shape:
                    image = cv2.resize(image, self.output_shape)
                all_images.append(image)
            calib = instant.calibs[c].crop(x_slice, y_slice)
            if self.rescale and self.output_shape:
                calib = calib.scale(*self.output_shape)
            if instant.download_flags & DownloadFlags.WITH_HUMAN_SEGMENTATION_MASKS and instant.human_masks:
                human_masks = instant.human_masks[c][y_slice, x_slice]
                if self.rescale and self.output_shape:
                    human_masks = cv2.resize(human_masks, self.output_shape)
            else:
                human_masks = None
        else:
            for offset in instant.offsets:
                # the coordinates of the rectangle below are probably wrong see documentation of cv2.rectangle
                raise NotImplementedError("TODO: check rectangle coordinates")
                image = cv2.rectangle(instant.all_images[(c,offset)], (x_slice.start, x_slice.stop), (y_slice.start, y_slice.stop), (255,0,0), 10)
                all_images.append(image)
            calib = instant.calibs[c]
            human_masks = instant.human_masks[c]

        return View(all_images, (x_slice, y_slice), calib, instant.annotations, rule_type=instant.rule_type, human_masks=human_masks, **kwargs)

    def augment(self, instant_key: InstantKey, instant: Instant):
        random_state = np.random.get_state()
        np.random.seed(instant_key.timestamp & 0xFFFFFFFF)

        for view_description in self.view_builder(instant_key, instant):
            view_key = ViewKey(instant_key, view_description.camera, view_description.index)
            yield view_key, self._crop_view(view_description, instant, **view_description.data)

        # required to keep the random seed random
        np.random.set_state(random_state)


# TODO: delete this 2 functions and use the Court or Calib objects
def get_court_keypoints(rule_type):
    w, h = court_dim[rule_type]
    return [Point3D(0,0,0), Point3D(w,0,0), Point3D(0,h,0), Point3D(w,h,0)]
def find_court_intersection_with_camera_border(calib, rule_type):
    def dichotomy(inside, outside, max_it=10):
        middle = Point3D((inside+outside)/2)
        if max_it == 0:
            return middle
        max_it = max_it - 1
        return dichotomy(middle, outside, max_it) if calib.projects_in(middle) else dichotomy(inside, middle, max_it)
    def find_point_inside(p1, p2, max_it=5):
        assert not calib.projects_in(p1) and not calib.projects_in(p2)
        middle = Point3D((p1+p2)/2)
        if calib.projects_in(middle):
            return middle
        if max_it == 0:
            return None
        point_inside = find_point_inside(middle, p2, max_it-1)
        if point_inside is not None:
            return point_inside
        return find_point_inside(middle, p1, max_it-1)

    points_inside = list()
    court_keypoints = get_court_keypoints(rule_type)

    for p1, p2 in zip(court_keypoints, court_keypoints[1:] + [court_keypoints[0]]):
        if calib.projects_in(p1) and calib.projects_in(p2):
            points_inside.append(p1)
            points_inside.append(p2)
        elif calib.projects_in(p1):
            points_inside.append(p1)
            points_inside.append(dichotomy(p1, p2))
        elif calib.projects_in(p2):
            points_inside.append(p2)
            points_inside.append(dichotomy(p2, p1))
        else:
            point_inside = find_point_inside(p1, p2)
            if point_inside is not None:
                points_inside.append(dichotomy(point_inside, p2))
                points_inside.append(dichotomy(point_inside, p1))

    return points_inside

