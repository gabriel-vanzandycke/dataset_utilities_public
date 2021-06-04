import struct
import urllib.request
from shapely.geometry import box, Polygon
import numpy as np
import cv2
from calib3d import Calib as Calib3d
from calib3d import Point3D, Point2D # pylint: disable=unused-import

class PointNotFoundException(Exception):
    pass

class Calib(Calib3d):
    @classmethod
    def from_dict(cls, dic):
        """ retro-compatibility for previous Calibration(DictObject) unpickling
        """
        dic.update(width=dic["img_width"], height=dic["img_height"], K=dic["KK"])
        return cls(**dic)

    def visible_edge(self, edge):
        def dichotomy(inside, outside, max_it=10):
            middle = Point3D((inside+outside)/2)
            if max_it == 0:
                return middle
            max_it = max_it - 1
            return dichotomy(middle, outside, max_it) if self.projects_in(middle) else dichotomy(inside, middle, max_it)
        def find_point_inside(p1, p2, max_it=4):
            assert not self.projects_in(p1) and not self.projects_in(p2)
            middle = Point3D((p1+p2)/2)
            if self.projects_in(middle):
                return middle
            if max_it == 0:
                return None
            point_inside = find_point_inside(middle, p2, max_it-1)
            if point_inside is not None:
                return point_inside
            return find_point_inside(middle, p1, max_it-1)

        p1, p2 = edge
        if self.projects_in(p1) and self.projects_in(p2):
            return p1, p2
        elif self.projects_in(p1):
            return p1, dichotomy(p1, p2)
        elif self.projects_in(p2):
            return dichotomy(p2, p1), p2
        else:
            point_inside = find_point_inside(p1, p2)
            if point_inside is None:
                raise ValueError
            return dichotomy(point_inside, p1), dichotomy(point_inside, p2)

    def to_basic_dict(self):
        return {
            "K": self.K,
            "r": cv2.Rodrigues(self.R)[0].flatten(),
            "T": self.T,
            "width": np.array([self.width]),
            "height": np.array([self.height]),
            "kc": np.array(self.kc),
        }

    @classmethod
    def parse_DeepSport(cls, data):
        return cls(**{
            "width": data["img_width"],
            "height": data["img_height"],
            "T": np.array([data["T"]]).T,
            "K": np.array(data["KK"]).reshape((3, 3)),
            "kc": np.array(data["kc"]),
            "R": np.array(data["R"]).reshape((3, 3))
        })

    @classmethod
    def parse_Keemotion(cls, arena_label, camera_index):
        url = "https://arena-data.keemotion.com/lookuptables/{}/TransitionView1.mat".format(arena_label)
        data = urllib.request.urlopen(url)
        version, nb_cam = struct.unpack("II", data.read(8))
        # We must loop through all the cameras because Calibs are dumped
        # successively in the file
        for _ in range(nb_cam):
            # pylint: disable=unused-variable
            cam_index = struct.unpack("I", data.read(4))
            cam_index = cam_index[0] if isinstance(cam_index, tuple) else cam_index
            K = np.array(struct.unpack("f"*9, data.read(4*9))).reshape((3, 3))
            R = np.array(struct.unpack("f"*9, data.read(4*9))).reshape((3, 3))
            Rc = struct.unpack("f"*9, data.read(4*9))
            T = np.array(struct.unpack("f"*3, data.read(4*3))).reshape((3, 1))
            kc = struct.unpack("f"*5, data.read(4*5))
            alpha = struct.unpack("f", data.read(4))
            _ = struct.unpack("f"*9*4, data.read(4*9*4))
            poscam = np.array(struct.unpack("f"*3, data.read(4*3)))
            width, height = struct.unpack("II", data.read(4*2))
            corners = struct.unpack("I"*2*4, data.read(4*2*4))
            w_distrort, h_distort = struct.unpack("I"*2, data.read(4*2))
            xshift_distort, yshift_distort = struct.unpack("i"*2, data.read(4*2))
            if cam_index == camera_index:
                return cls(width=width, height=height, T=T, K=K, R=R, kc=kc)
        raise ValueError("Impossible to find camera #{} for arena {}. nb_cam read={}. version={}".format(camera_index,
                                                                                                     arena_label, nb_cam, version))

    def get_region_visible_corners_2d(self, points_3d: Point3D, approximate_curve_by_N_segments=10):
        """Return a list of corner points defining the 2D boundaries of a specific 3D region on the image space

        Args:
            points_3d ([type]): [description]
            approximate_curve_by_N_segments (int, optional): [description]. Defaults to 10.

        Returns:
            List[Tuple(int, int)]: a list of 2D coordinates of the corner points of a specific 3D region on the image space
        """

        # Construct the polygon defining the boundaries of the 3D region and projects it, considering the lens distorsion (3D straight lines might be curves on the image)
        region_3d_coords = points_3d.close().linspace(approximate_curve_by_N_segments+1)
        region_2d_coords = self.project_3D_to_2D(region_3d_coords)
        any_coord_outside_img_boundaries = np.any(region_2d_coords < 0) or \
                                           np.any(region_2d_coords.x >= self.width) or \
                                           np.any(region_2d_coords.y >= self.height)
        if not any_coord_outside_img_boundaries:
            return region_2d_coords

        # Restrict the 2D region polygon to the image space boundaries
        img_corners = box(minx=0, miny=0, maxx=self.width, maxy=self.height)
        region_corners = Polygon([r.to_int_tuple() for r in region_2d_coords])
        region_polygon_restricted_to_img_space = region_corners.intersection(img_corners)

        if region_polygon_restricted_to_img_space:
            return Point2D(np.array(region_polygon_restricted_to_img_space.exterior.coords).T)
        else:
            return Point2D(np.empty(shape=(2, 0), dtype=float))

try:
    from mako.template import Template # pylint: disable=import-error

    class CalibCuda(Calib):
        datatypes = {
                'width':  ('__align__(8) int',    '',     8),
                'height': ('__align__(8) int',    '',     8),
                'T':      ('__align__(8) double', '[3]',  24),
                'K':      ('__align__(8) double', '[9]',  72),
                'kc':     ('__align__(8) double', '[5]',  40),
                'R':      ('__align__(8) double', '[9]' , 72),
                'poscam': ('__align__(8) double', '[3]' , 24),
                'P':      ('__align__(8) double', '[12]', 96),
                'Pinv':   ('__align__(8) double', '[12]', 96),
                'Kinv':   ('__align__(8) double', '[9]',  72)
            }

        @classmethod
        def struct_str(cls):
            return Template("""
                struct calib_t
                {
                    %for var in data:
                    ${var[1]} ${var[0]}${var[2]};
                    %endfor
                };
            """).render(data=[(name, dtype, size) for name, (dtype, size, _) in  cls.datatypes.items()])

        @classmethod
        def memsize(cls):
            return sum([size for attr, (_, _, size) in cls.datatypes.items()])

        @staticmethod
        def from_calib(calib):
            new_calib = CalibCuda.__new__(CalibCuda)
            new_calib.update(**calib.dict)
            return new_calib

        def memset(self, ptr, callback):
            # check function 'pycuda.tools.dtype_to_ctype'
            offset = 0
            for name, (_, _, size) in self.datatypes.items():
                value = self.__dict__[name]
                if isinstance(value, int):
                    data = memoryview(np.int32(value))
                elif isinstance(value, float):
                    data = memoryview(np.float32(value))
                else:
                    data = memoryview(value)
                callback(int(ptr)+offset, data)
                offset += size
except ModuleNotFoundError as e:
    if e.name == "mako":
        pass

class ProjectiveDrawer():
    def __init__(self, calib, color, thickness=1, segments=10):
        self.calib = calib
        self.color = color
        self.thickness = thickness
        self.segments = segments

    def draw_line(self, image, point3D1, point3D2):
        try:
            point3D1, point3D2 = self.calib.visible_edge((point3D1, point3D2))
        except ValueError:
            return
        points3D = Point3D(np.linspace(point3D1, point3D2, self.segments+1))
        points2D = self.calib.project_3D_to_2D(points3D).astype(np.int32)
        cv2.polylines(image, [points2D.T.reshape(-1, 1, 2)], False, color=self.color, thickness=self.thickness)

    def draw_arc(self, image, center, radius, start_angle=0.0, stop_angle=2*np.pi):
        angles = np.linspace(start_angle, stop_angle, self.segments*4+1)
        xs = np.cos(angles)*radius + center.x
        ys = np.sin(angles)*radius + center.y
        zs = np.ones_like(angles)*center.z
        points3D = Point3D(np.vstack((xs,ys,zs)))
        points2D = self.calib.project_3D_to_2D(points3D).astype(np.int32).T.reshape((-1,1,2))
        if self.thickness == -1:
            cv2.fillPoly(image, [points2D], color=self.color)
        else:
            cv2.polylines(image, [points2D], False, color=self.color, thickness=self.thickness)

    def draw_rectangle(self, image, point3D1, point3D2):
        c1 = point3D1
        c3 = point3D2
        if point3D1.z == point3D2.z:
            c2 = Point3D(c1.x, c3.y, c1.z)
            c4 = Point3D(c3.x, c1.y, c1.z)
        elif point3D1.x == point3D2.x:
            c2 = Point3D(c1.x, c1.y, c3.z)
            c4 = Point3D(c1.x, c3.y, c1.z)
        elif point3D1.y == point3D2.y:
            c2 = Point3D(c1.x, c1.y, c3.z)
            c4 = Point3D(c3.x, c1.y, c1.z)
        corners = [c1, c2, c3, c4, c1]
        for p1, p2 in zip(corners, corners[1:]):
            self.draw_line(image, p1, p2)

    def fill_polygon(self, image, points3D):
        points3D = points3D.close().linspace(self.segments)
        points2D = self.calib.project_3D_to_2D(points3D).astype(np.int32)
        cv2.fillPoly(image, points2D.T[np.newaxis], color=self.color)

def line_plane_intersection(plane_normal, plane_point, line_direction, line_point, epsilon=1e-5):
    dot = np.dot(plane_normal.T, line_direction)
    if np.abs(dot) < epsilon:
        return None
    w = np.subtract(line_point, plane_point)
    factor = -np.dot(plane_normal.T, w) / dot
    return np.add(w, np.add(factor*line_direction, plane_point))

def compute_length3D(point2D, point3D, calib: Calib):
    plane_normal = np.subtract(calib.C, point3D)
    line_direction = np.subtract(calib.C, calib.project_2D_to_3D(point2D, Z=0))
    intersection = line_plane_intersection(plane_normal, point3D, line_direction, calib.C)
    return np.linalg.norm(point3D - intersection)

def rescale_and_recenter(H, nx, ny):
    # TODO: only consider points on the court and 2m above
    # => it's the only usefull area on the recentered and rescaled image

    # See where image corners get displaced
    p = H @ np.transpose(np.array([[0,0,1],[nx,0,1],[nx,ny,1],[0,ny,1]]))

    # Handle projection inversions:
    # Trick (to address issue in KS-US-MARQUETTEBIRD):
    # if inversion occurs, just set point at double the size of the image (because it is just impossible
    # to go up to infinity. This means we will discard all pixels coresponding to the ground below the
    # camera, and also all pixels further below.)
    pnorm=np.zeros((3,4))
    for i in range(4):
        if p[2,i] > 0: # no inversion
            pnorm[:,i] = p[:,i]/p[2,i]
        else: # inversion took place
            ptemp=p[:,i]/p[2,i]
            ptemp[0] = 2*nx if ptemp[0] < 0 else -nx
            ptemp[1] = 3*ny if ptemp[1] < 0 else -ny
            pnorm[:,i] = ptemp

    minx, maxx = np.min(pnorm[0,:]), np.max(pnorm[0,:])
    miny, maxy = np.min(pnorm[1,:]), np.max(pnorm[1,:])

    # Compute scaling matrix that should be applied to keep corners inside the image
    Hscale=np.array([[nx/(maxx-minx),0,1],[0,ny/(maxy-miny),1],[0,0,1]])

    # See where new corners get displaced after rescale
    p = Hscale @ pnorm

    # Handle projection inversions again
    pnorm = np.zeros((3,4))
    for i in range(4):#=1:4
        if p[2,i] > 0: # no inversion
            pnorm[:,i] = p[:,i]/p[2,i]
        else:
            ptemp=p[:,i]/p[2,i]
            ptemp[0] = 2*nx if ptemp[0] < 0 else -nx
            ptemp[1] = 2*ny if ptemp[1] < 0 else -ny
            pnorm[:,i] = ptemp

    minx=np.min(pnorm[0,:])
    miny=np.min(pnorm[1,:])
    Hshift=np.array([[1,0,-minx],[0,1,-miny],[0,0,1]])

    # Combine scaling and shift
    return Hshift@Hscale

def set_z_vanishing_point(P, nx, ny):
    # Compute the rotation that should be applied to have the
    # projection of somebody standing at the center of the image
    # parralel to vertical

    # 2D position of someone at the center of the image
    center = np.array([nx/2,ny/2,1])
    Hg = P[:,[0,1,3]]
    iHg = np.linalg.inv(Hg)
    pos_center = np.matmul(iHg,np.transpose(center))
    pos_center = pos_center/pos_center[2]

    # Projection of the head (180cm) of the person on the image.
    pos_head=np.array([pos_center[0], pos_center[1], -180, 1])
    pos_head_im=P@pos_head
    pos_head_im=pos_head_im/pos_head_im[2]

    # Correction angle
    alpha=np.arctan2(nx/2-pos_head_im[0],ny/2-pos_head_im[1])
    alpha_degres = alpha*180/np.pi # pylint: disable=unused-variable

    # Corrected P matrix (rotation around the center of the image)
    Tmat=np.array([[1,0,nx/2],[0,1,ny/2],[0,0,1]])
    Rot=[[np.cos(alpha),-np.sin(alpha),0],[np.sin(alpha),np.cos(alpha),0],[0,0,1]]
    Rotmat=Tmat@Rot@np.linalg.inv(Tmat)
    P=Rotmat@P

    # Set z vanishing point at infinity
    fact1=P[0,2]/P[1,2]
    fact2=P[2,2]/P[1,2]
    Mat_vert=np.array([[1,-fact1,0],[0,1,0],[0,-fact2,1]])
    newP=Mat_vert@P

    # Compute shear (persons on the same line in the image should have
    # same size projections)
    K=newP[2,0]/newP[2,1]
    shearf=(K*newP[1,1]-newP[1,0])/(newP[0,0]-K*newP[0,1])
    shear=[[1,0,0],[shearf,1,0],[0,0,1]]

    # Rescale and recenter to keep all pixels inside the transformed image
    SSmat=rescale_and_recenter(shear@Mat_vert@Rotmat, nx, ny)

    # Combined transformation
    return SSmat@shear@Mat_vert@Rotmat

def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if width > image_size[0]:
        width = image_size[0]

    if height > image_size[1]:
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]

def find_angles(P):
    """ Computes Euler angles towards point P """
    norm = lambda x: x/np.linalg.norm(x)
    newy = norm(np.cross(P.T[0], Point3D(1,0,0).T[0]))
    rotx = np.arccos(np.dot(newy, Point3D(0,1,0).T[0]))
    newz = norm(np.cross(Point3D(1,0,0).T[0], newy))
    roty = np.arccos(min(1.0,np.dot(newz, norm(P.T[0]))))
    newx = norm(np.cross(newy, norm(P.T[0])))
    rotz = np.arccos(np.dot(newx, np.array([newx[0], newx[1], 0])))
    return (rotx, roty, rotz)
class PanoramicStitcher():
    def __init__(self, calibs, output_shape):

        w, h = output_shape
        K = np.array([[f ,   0  , w/2 ], [  0  , f , h/2 ], [  0  ,   0  ,  1  ]])
        C = calibs[0].C
        R = cv2.Rodrigues(np.array(find_angles(Point3D(1400,600,0)-C)))[0]
        T = -R@C
        camera = Calib(K=K, T=T, R=R, width=w, height=h)
        self.camera = camera
        self.calibs = calibs

        w, h = camera.width, camera.height
        self.w, self.h = w, h
        points2D = Point2D(np.stack(np.meshgrid(np.arange(w),np.arange(h))).reshape((2,w*h)))
        points3D = camera.project_2D_to_3D(points2D, 0)

        points2D = calibs[0].project_3D_to_2D(points3D).astype(np.int32)
        indices = np.where(np.logical_or(np.any(points2D < Point2D(0,0), axis=0), np.any(points2D >= Point2D(calibs[0].width, calibs[0].height), axis=0)))[0]
        points2D[:,indices] = 0
        self.points2D0 = points2D

        points2D = calibs[1].project_3D_to_2D(points3D).astype(np.int32)
        indices = np.where(np.logical_or(np.any(points2D < Point2D(0,0), axis=0), np.any(points2D >= Point2D(calibs[1].width, calibs[1].height), axis=0)))[0]
        points2D[:,indices] = 0
        self.points2D1 = points2D

    def __call__(self, images):
        image_0 = images[0][self.points2D0.y, self.points2D0.x].reshape(self.h,self.w,3)
        image_1 = images[1][self.points2D1.y, self.points2D1.x].reshape(self.h,self.w,3)
        s = np.stack((image_0, image_1))
        return np.max(s, axis=0)