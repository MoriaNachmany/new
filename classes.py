from dataclasses import dataclass
import numpy as np
import math
from functools import partial
ANGLE = 0.01

p = partial(np.linalg.norm, ord=2)

@dataclass
class Ray:
    origin:    np.ndarray
    direction: np.ndarray

    def __post_init__(self):
        self.direction /= p(self.direction)

    def project(self, point):
        return np.dot(self.direction, point - self.origin)

    def get_point_at_distance(self, t):
        return self.origin + t * self.direction

@dataclass
class Camera:
    position:     np.ndarray
    look_at:      np.ndarray
    up_vector:           np.ndarray
    screen_distance:  float
    screen_width: float
    screen_height: float


@dataclass
class Set:
    background_color:   np.ndarray
    shadow_rays: int
    max_depth:   int


@dataclass
class Material:
    diffuse_color:  np.ndarray
    specular_color: np.ndarray
    reflection_color:  np.ndarray
    phong:        float
    transparency:       float


@dataclass
class Light:
    position:        np.ndarray
    color:             np.ndarray
    specular_intensity: float
    shadow_intensity:   float
    radius:          float


@dataclass
class Shape:
    material: int

    def find_intersection(self, ray: Ray):
        raise NotImplementedError(f'the subclass {self.__class__} did not implement this method')

    def normal_at_point(self, point: np.array):
        raise NotImplementedError(f'the subclass {self.__class__} did not implement this method')


@dataclass
class Sphere(Shape):
    center: np.ndarray
    radius: float

    def find_intersection(self, ray: Ray):
        center = self.center
        radius = self.radius
        object_len = ray.project(center)
        if (object_len <= 0):
            return False
        object_center = center - ray.origin
        object_len = p(object_center)
        len = np.sqrt(math.pow(object_len, 2) - math.pow(object_len, 2))
        if np.isclose(len, radius, rtol=ANGLE):
            return ray.get_point_at_distance(object_len)
        if (len > radius):
            return False
        abs = np.sqrt(math.pow(radius, 2) - math.pow(len, 2))
        p1 = ray.get_point_at_distance(object_len - abs)
        p2 = ray.get_point_at_distance(object_len + abs)
        if (np.linalg.norm(ray.origin-p1) < np.linalg.norm(ray.origin-p2)):
            return p1
        return p2

    def normal_at_point(self, point: np.array):
        vector_normal = point-self.center
        vector_normal = vector_normal/np.linalg.norm(vector_normal)
        return vector_normal

@dataclass
class Plane(Shape):
    normal: np.ndarray
    offset: float

    def __post_init__(self):
        self.normal /= p(self.normal)

    def find_intersection(self, ray: Ray):

        calc_dot = np.dot(self.normal, ray.direction)
        if np.isclose(calc_dot, 0, rtol=0.01):
            return False
        dot_output = np.dot(self.normal, ray.origin)
        t = (self.offset - dot_output) / calc_dot
        point = ray.get_point_at_distance(t)
        return point

    def normal_at_point(self, point: np.array):
        return self.normal

@dataclass
class Box(Shape):
    center: np.ndarray
    length: float
    min: np.ndarray
    max: np.ndarray

    def find_intersection(self, ray: Ray):
        box_min = self.center - self.length / 2
        box_max = self.center + self.length / 2
        t_min = np.divide(box_min - ray.origin, ray.direction)
        t_max = np.divide(box_max - ray.origin, ray.direction)
        if t_min[0] > min(t_max[1], t_max[2]) | t_min[1] > min(t_max[0], t_max[2]) | t_min[2] > min(t_max[0], t_max[1]):
            return False
        t = t_min.max()
        return ray.get_point_at_distance(t)

    def normal_at_point(self, point:np.array):
        normal = np.zeros(3)
        if point[0] == self.min[0]:
            normal += [-1, 0, 0]
        elif point[0] == self.max[0]:
            normal += [1, 0, 0]
        if point[1] == self.min[1]:
            normal += [0, -1, 0]
        elif point[1] == self.max[1]:
            normal += [0, 1, 0]
        if point[2] == self.min[2]:
            normal += [0, 0, -1]
        elif point[2] == self.max[2]:
            normal += [0, 0, 1]
        return normal / np.linalg.norm(normal)
