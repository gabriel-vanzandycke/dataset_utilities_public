import random
import os

import cv2
import numpy as np

from dataset_utilities.calib import set_z_vanishing_point, Point3D, Point2D
from dataset_utilities.court import Court, BALL_DIAMETER
from dataset_utilities.transforms import Transform
from dataset_utilities.utils import gamma_correction, RandomCropper, IncompatibleCropException, setdefaultattr, parameters_to_affine_transform
from .views_dataset import View, ViewKey

try:
    from dataset_utilities.calib import CalibrationCuda
    import pycuda.driver as cuda
    # pylint: disable=unused-import
    import pycuda.autoinit
    # pylint: enable=unused-import
    from pycuda.compiler import SourceModule

    class AddBallDistance(Transform):
        def __init__(self):
            self._calib_struct_ptr = cuda.mem_alloc(CalibrationCuda.memsize())
            self._ball_ptr = cuda.mem_alloc(3*8)
            cuda_code = open(os.path.join(os.path.dirname(__file__), "mod_source.c"), "r").read()
            mod = SourceModule(str(CalibrationCuda.struct_str())+cuda_code)
            self._ball_distance = mod.get_function("BallDistance")
            self._bdim = (32,32,1)

        def __repr__(self):
            return "{}()".format(self.__class__.__name__)

        def __call__(self, key, view: View):
            # copy calib to GPU
            calib = CalibrationCuda.from_calib(view.calib)
            calib.memset(self._calib_struct_ptr, cuda.memcpy_htod)

            # copy ball position to GPU
            cuda.memcpy_htod(self._ball_ptr, memoryview(view.ball.center))

            # create distmap on GPU
            distmap_gpu = cuda.mem_alloc(calib.img_width * calib.img_height * 8)# 8 bytes per double
            cuda.memset_d8(distmap_gpu, 0, calib.img_width * calib.img_height * 8)

            # compute best block and grid dimensions
            dx, mx = divmod(calib.img_width, self._bdim[0])
            dy, my = divmod(calib.img_height, self._bdim[1])
            gdim = ( (dx + (mx>0)) * self._bdim[0], (dy + (my>0)) * self._bdim[1])

            # call gpu function
            self._ball_distance(distmap_gpu, self._calib_struct_ptr, self._ball_ptr, block=self._bdim, grid=gdim)

            # copy result to memory
            view.ball_distance = np.zeros((calib.img_height,calib.img_width))#, np.int8)
            cuda.memcpy_dtoh(view.ball_distance, distmap_gpu)
            # cuda.Context.synchronize()
            return view
except ModuleNotFoundError as e:
    if e.name != "cuda":
        raise e

except ImportError as e:
    if "CalibrationCuda" not in str(e.msg):
        raise e

class AddBallAnnotation(Transform):
    def __call__(self, key, view):
        view.ball = [a for a in view.annotations if a.type == 'ball'][0]
        return view

class UndistortTransform(Transform):
    def __call__(self, key, view):
        all_images = []
        for image in view.all_images:
            all_images.append(cv2.undistort(image, view.calib.K, view.calib.kc))
        calib = view.calib.update(kc=np.array([0,0,0,0,0]))
        return View(all_images, view.box, calib, view.annotations)

class RectifyTransform(Transform):
    def __init__(self, output_shape=None):
        self.output_shape = output_shape

    def __call__(self, key, view: View):
        output_shape = (view.image.shape[1], view.image.shape[0]) if self.output_shape is None else self.output_shape
        H = set_z_vanishing_point(view.calib.P, view.calib.img_width, view.calib.img_height)
        all_images = []
        for image in view.all_images:
            all_images.append(cv2.warpPerspective(image, H, output_shape))
        calib = view.calib.update(K=H@view.calib.K)
        return View(all_images, view.box, calib, view.annotations)

class RectifyUndistortTransform(Transform):
    def __init__(self, output_shape=None, interpolation=cv2.INTER_CUBIC):
        self._lookuptables = dict()
        self.output_shape = output_shape
        self.interpolation = interpolation

    def _get_lookuptables(self, arena_label, cam_idx, calib, input_shape):
        cam_key = (arena_label, cam_idx)
        if cam_key not in self._lookuptables:
            # Rescale transform
            output_shape = self.output_shape if self.output_shape else input_shape
            sx = output_shape[0]/input_shape[0]
            sy = output_shape[1]/input_shape[1]
            S = np.array([[sx,0 ,0],
                          [0 ,sy,0],
                          [0 ,0 ,1]])

            # Undistort transform
            map1, map2 = cv2.initUndistortRectifyMap(calib.K, calib.kc, None, S@calib.K, output_shape, cv2.CV_32FC1)
            calib = calib.update(kc=np.array([0,0,0,0,0]), K=S@calib.K, width=output_shape[0], height=output_shape[1])

            # Rectify transform
            H = set_z_vanishing_point(calib.P, output_shape[0], output_shape[1])
            calib = calib.update(K=H@calib.K, kc=np.array([0,0,0,0,0]))
            input_map = np.stack((map1,map2,np.ones_like(map1)), axis=2)
            output_map = np.tensordot(np.linalg.inv(S)@np.linalg.inv(H)@S, input_map, axes=((1),(2))).astype(np.float32)
            map1, map2 = cv2.convertMaps(output_map[0]/output_map[2], output_map[1]/output_map[2], cv2.CV_16SC2) # convertMaps to speed up the lookup

            # Store lookup table
            self._lookuptables[cam_key] = (calib, map1, map2)
        return self._lookuptables[cam_key]

    def __call__(self, view_key: ViewKey, view: View):
        calib, map1, map2 = self._get_lookuptables(view_key.instant_key.arena_label, view_key.camera, view.calib, (view.image.shape[1], view.image.shape[0]))
        all_images = []
        for image in view.all_images:
            all_images.append(cv2.remap(image, map1, map2, self.interpolation))
        return View(all_images, view.box, calib, view.annotations)

class ComputeDiff(Transform):
    def __init__(self, squash=False, inplace=False):
        self.squash = squash
        self.inplace = inplace

    def __call__(self, view_key: ViewKey, view: View):
        diff = np.abs(view.image.astype(np.int32) - view.all_images[1].astype(np.int32)).astype(np.uint8)
        if self.squash:
            diff = np.mean(diff, axis=2).astype(np.uint8)
        if self.inplace:
            view.image = np.dstack((view.image, diff))
        else:
            view.diff = diff
        return view


class GameGammaColorTransform(Transform):
    def __init__(self, transform_dict):
        assert all([isinstance(k, int) for k in transform_dict.keys()])
        #29582 : [1.04, 1.02, 0.93],
        #24651 : [1.05, 1.02, 0.92],
        #30046 : [1.01, 1.01, 1.01]
        self.transform_dict = transform_dict

    def __call__(self, view_key, view):
        if view_key.instant_key.game_id in self.transform_dict.keys():
            gammas = np.array(self.transform_dict[view_key.instant_key.game_id])
            view.image = gamma_correction(view.image, gammas)
        return view




class BayeringTransform(Transform):
    def __init__(self):
        self.R_filter = np.array([[1,0],[0,0]])
        self.G_filter = np.array([[0,1],[1,0]])
        self.B_filter = np.array([[0,0],[0,1]])
    def __call__(self, view_key, view):
        height, width, _ = view.image.shape
        R_mask = np.tile(self.R_filter, [height//2, width//2])
        G_mask = np.tile(self.G_filter, [height//2, width//2])
        B_mask = np.tile(self.B_filter, [height//2, width//2])
        mask = np.stack((R_mask, G_mask, B_mask), axis=2)
        mask = mask[np.newaxis]
        for i, image in enumerate(view.all_images):
            view.all_images[i] = np.sum(image*mask, axis=3)
        view.image = view.all_images[0]


class GameRGBColorTransform(Transform):
    def __init__(self, transform_dict):
        assert all([isinstance(k, int) for k in transform_dict.keys()])
        self.transform_dict = transform_dict

    def __call__(self, view_key: ViewKey, view: View):
        if view_key.instant_key.game_id in self.transform_dict.keys():
            adaptation_vector = np.array(self.transform_dict[view_key.instant_key.game_id])
            view.image = np.clip(view.image.astype(np.float32)*adaptation_vector, 0, 255).astype(np.uint8)
        return view

class ViewCropperTransform(Transform):
    def __init__(self, output_shape, def_min=60, def_max=160, max_angle=8, do_flip=False, focus_object=None, debug=False):
        """
            def -  definition in pixels per meters. 60px/m = ball of 14px
        """
        self.output_shape = output_shape
        self.do_flip = do_flip
        assert not self.do_flip, "There seem to be a bug in the flip"
        self.random_cropper = RandomCropper(output_shape, def_min=def_min, def_max=def_max, margin=200, max_angle=max_angle)
        self.focus_object = focus_object
        self.debug = debug
        self.__keys_counter = {}
        self.focus_keypoints_functions = {
            "ball": self.focus_on_ball,
            "player": self.focus_on_player,
            None: self.focus_random,
        }
    @property
    def config(self):
        return {
            "output_shape": self.output_shape,
            "do_flip": self.do_flip,
            "focus_object": self.focus_object, "debug": self.debug,
            **self.random_cropper.config
        }
    def focus_on_ball(self, view_key, view):
        balls = [a for a in view.annotations if a.type == "ball" and a.camera == view_key.camera]
        if not balls:
            return None
        ball = random.sample(balls, 1)[0]
        return ball.center
    def focus_on_player(self, view_key, view):
        players = [a for a in view.annotations if a.type == "player" and a.camera == view_key.camera]
        if not players:
            return None
        player = random.sample(players, 1)[0]
        keypoints = Point3D([player.head, player.hips, player.foot1, player.foot2])
        return keypoints
    def focus_random(self, view_key, view):
        court = setdefaultattr(view, "court", Court(getattr(view, "rule_type", "FIBA")))
        top_edge = list(court.visible_edges(view.calib))[0]
        start = top_edge[0][0][0]
        stop = top_edge[1][0][0]
        x = np.random.beta(2, 2)*(stop-start)+start
        y = np.random.beta(2, 2)*court.h/2+court.h/4
        z = 0
        return Point3D(x,y,z)

    def reset_seed(self):
        self.__keys_counter = {}

    def __call__(self, view_key: ViewKey, view: View):
        self.__keys_counter[view_key] = self.__keys_counter.get(view_key, -1) + 1

        # Set random seed with timestamp
        random_state = np.random.get_state()
        np.random.seed(self.__keys_counter[view_key])
        seed = np.random.randint(0, 10000)

        keypoints = self.focus_keypoints_functions[self.focus_object](view_key, view)
        if keypoints is None:
            return None

        np.random.set_state(random_state)
        try:
            angle, x_slice, y_slice = self.random_cropper(view.calib, keypoints, seed)
        except IncompatibleCropException:
            return None

        A = parameters_to_affine_transform(angle, x_slice, y_slice, self.output_shape, self.do_flip)

        if self.debug:
            w, h = self.output_shape
            points = Point2D(np.linalg.inv(A)@Point2D([Point2D(0,0), Point2D(0,h), Point2D(w, h), Point2D(w,0)]).H)
            cv2.polylines(view.image, [points.T.astype(np.int32)], True, (255,0,0), 4)
        else:
            view.image = cv2.warpAffine(view.image, A[0:2,:], self.output_shape, flags=cv2.INTER_LINEAR)
            view.calib = view.calib.update(K=A@view.calib.K, width=self.output_shape[0], height=self.output_shape[1])

            if hasattr(view, "all_images"):
                for i in range(1, len(view.all_images)):
                    view.all_images[i] = cv2.warpAffine(view.all_images[i], A[0:2,:], self.output_shape, flags=cv2.INTER_LINEAR)
            if hasattr(view, "human_masks") and view.human_masks is not None:
                view.human_masks = cv2.warpAffine(view.human_masks, A[0:2,:], self.output_shape, flags=cv2.INTER_NEAREST)

        return view

class ExtractViewData(Transform):
    def __init__(self, *factories):
        self.factories = factories
    def __call__(self, view_key, view):
        if not view:
            return None
        data = {"input_image": view.image}
        for factory in self.factories:
            if factory is None:
                continue
            try:
                data.update(**factory(view_key, view))
            except:
                print(factory)
                raise
        return data

class AddDiffFactory(Transform):
    def __call__(self, view_key, view):
        return {"input_image2": view.all_images[1]}

class AddCalibFactory(Transform):
    def __call__(self, view_key, view):
        return {"calib": view.calib, **view.calib.to_basic_dict()}

class AddCourtFactory(Transform):
    def __call__(self, view_key, view):
        if not getattr(view, "court", None):
            view.court = Court()
        return {
            "court_width": np.array([view.court.w]),
            "court_height": np.array([view.court.h])
        }

class AddHumansSegmentationTargetViewFactory(Transform):
    def __call__(self, view_key, view):
        return {"human_masks": view.human_masks}

class AddBallSegmentationTargetViewFactory(Transform):
    def __call__(self, view_key, view):
        calib = view.calib
        target = np.zeros((calib.height, calib.width), dtype=np.uint8)
        for ball in [a for a in view.annotations if a.type == "ball" and calib.projects_in(a.center) and a.visible]:
            diameter = calib.compute_length2D(BALL_DIAMETER, ball.center)
            center = calib.project_3D_to_2D(ball.center)
            cv2.circle(target, center.to_int_tuple(), radius=int(diameter/2), color=1, thickness=-1)
        return {
            "target": target
        }
