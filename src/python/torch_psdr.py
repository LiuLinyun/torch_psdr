from operator import iconcat
from turtle import forward
from typing import Union, List, Tuple
import gc


import torch
from torch import Tensor, FloatTensor, IntTensor, LongTensor, norm
from torch.autograd import Function as FunctionD

import enoki as ek
import enoki.cuda
import enoki.cuda_autodiff
from enoki.cuda import (
    Float32 as FloatC,
    Vector3f as Vec3fC,
    Vector3i as Vec3iC,
    Vector2f as Vec2fC,
)
from enoki.cuda_autodiff import (
    Float32 as FloatD,
    Vector3f as Vec3fD,
    Vector2f as Vec2fD,
)

import pypsdr


class Wrap2torch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        # args: [integrator_, scene_, *enoki_vars, *torch_vars]
        integrator_ = args[0]
        scene_ = args[1]
        vars_cnt = (len(args) - 2) // 2
        enoki_vars = list(args[2 : 2 + vars_cnt])
        ctx.input = enoki_vars
        imgs = integrator_.renderD(scene_)
        ctx.out = imgs
        out_torchs = [img.torch() for img in imgs]
        for t in range(len(out_torchs)):
            out_torchs[t].requires_grad_(True)
        out_torchs = tuple(out_torchs)
        return out_torchs

    # @torch.autograd.function.once_differentiable
    @staticmethod
    def backward(ctx, *grads):
        for i in range(len(ctx.out)):
            if grads[i] is not None and ek.requires_gradient(ctx.out[i]):
                if isinstance(ctx.out[i], FloatD):
                    ek.set_gradient(ctx.out[i], FloatC(grads[i]))
                elif isinstance(ctx.out[i], Vec3fD):
                    ek.set_gradient(ctx.out[i], Vec3fC(grads[i]))
                else:
                    print("Invalid data type")
        FloatD.backward()
        vars_cnt = len(ctx.input)
        out_grad = (
            (
                None,
                None,
            )
            + (None,) * vars_cnt
            + tuple(
                ek.gradient(ctx.input[i]).torch()
                if ek.requires_gradient(ctx.input[i])
                else None
                for i in range(vars_cnt)
            )
        )
        return out_grad


def torch2enoki(val: FloatTensor) -> Union[FloatD, Vec3fD]:
    if val.dim() == 1:
        out = FloatD(val)
    elif val.dim() == 2 and val.size(-1) == 3:
        out = Vec3fD(val)
    ek.set_requires_gradient(out, val.requires_grad)
    return out


class VarContex:
    def __init__(self):
        self.enoki_vars = []
        self.torch_vars = []

    def append_variable(self, torch_var: FloatTensor, enoki_var: Union[FloatD, Vec3fD]):
        if torch_var.requires_grad:
            self.enoki_vars.append(enoki_var)
            self.torch_vars.append(torch_var)

    def get_variables(self):
        return self.enoki_vars, self.torch_vars

    def merge_variables(
        self, enoki_vars: List[Union[FloatD, Vec3fD]], torch_vars: FloatTensor
    ):
        for e, t in zip(enoki_vars, torch_vars):
            if t.requires_grad:
                self.enoki_vars.append(e)
                self.torch_vars.append(t)


class BaseCamera(VarContex):
    """This is a base camera class for PerspectiveCamera and OrthographicCamera"""

    def __init__(self):
        super().__init__()


class PerspectiveCamera(BaseCamera):
    def __init__(
        self,
        look_from: FloatTensor,
        look_at: FloatTensor,
        up: FloatTensor,
        vfov: FloatTensor,
        height: int,
        width: int,
    ):
        """Constructor for PerspectiveCamera

        Args:
            look_from (FloatTensor): shape (3,), camera position
            look_at (FloatTensor): shape (3,), the point where the camera look at
            up (FloatTensor): shape (3,), the up direction of camera
            vfov (FloatTensor): shape(1,), vertical field-of-view angle in rad
            height (int): rendered image height
            width (int): rendered image width
        """
        super().__init__()
        assert look_from.size() == (3,)
        assert look_at.size() == (3,)
        assert up.size() == (3,)
        assert vfov.size() == (1,)
        assert height > 0 and width > 0

        look_from = look_from.unsqueeze(0)
        look_at = look_at.unsqueeze(0)
        up = up.unsqueeze(0)

        look_from_ = torch2enoki(look_from)
        look_at_ = torch2enoki(look_at)
        up_ = torch2enoki(up)
        vfov_ = torch2enoki(vfov)

        self.append_variable(look_from, look_from_)
        self.append_variable(look_at, look_at_)
        self.append_variable(up, up_)
        self.append_variable(vfov, vfov_)

        self.height = height
        self.width = width
        self.camera_ = pypsdr.PerspectiveCamera(
            look_from_, look_at_, up_, vfov_, height, width
        )


class OrthographicCamera(BaseCamera):
    def __init__(
        self,
        look_from: FloatTensor,
        look_at: FloatTensor,
        up: FloatTensor,
        view_width: FloatTensor,
        height: int,
        width: int,
    ):
        """Constructor for OrthographicCamera

        Args:
            look_from (FloatTensor): shape (3,), camera position
            look_at (FloatTensor): shape (3,), the point where the camera look at
            up (FloatTensor): shape (3,), the up direction of camera
            view_width (FloatTensor): shape (1,), visible width of camera in world coordinate space
            height (int): rendered image height
            width (int): rendered image width
        """
        super().__init__()
        assert look_from.size() == (3,)
        assert look_at.size() == (3,)
        assert up.size() == (3,)
        assert view_width.size() == (1,)
        assert height > 0 and width > 0

        look_from = look_from.unsqueeze(0)
        look_at = look_at.unsqueeze(0)
        up = up.unsqueeze(0)

        look_from_ = torch2enoki(look_from)
        look_at_ = torch2enoki(look_at)
        up_ = torch2enoki(up)
        view_width_ = torch2enoki(view_width)

        self.append_variable(look_from, look_from_)
        self.append_variable(look_at, look_at_)
        self.append_variable(up, up_)
        self.append_variable(view_width, view_width_)

        self.height = height
        self.width = width
        self.camera_ = pypsdr.OrthographicCamera(
            look_from_, look_at_, up_, view_width_, height, width
        )


class Texture(VarContex):
    def __init__(self, tex: FloatTensor):
        """
        Texture is represented as a constant texture value or texture map per mesh

        Args:
            tex (FloatTensor): texture data for mesh. This can be a tensor of shape (1,) or (3,) for constant texture
                of 1 or 3 channel(s) texture, or a tensor of shape (H, W, 1) or (H, W, 3) for image based 1 or 3 channel(s) texture map,
                or (H, W) for 1 channel texture map
        """
        super().__init__()
        tex_type = None
        tex_size = tex.size()
        tex_dims = tex.dim()
        tex_type = None
        if tex_dims == 1:
            if tex_size == (1,):
                tex_type = "Constant1f"
            elif tex_size == (3,):
                tex_type = "Constant3f"
                tex = tex.unsqueeze(0)
            else:
                pass
        elif tex_dims == 2:
            tex_type == "UV1f"
            tex = tex.unsqueeze(-1)
        elif tex_dims == 3:
            if tex_size[-1] == 1:
                tex_type = "UV1f"
            elif tex_size[-1] == 3:
                tex_type = "UV3f"
            else:
                pass
        else:
            pass

        assert tex_type != None
        self.tex_type = tex_type

        if tex_type == "UV1f" or tex_type == "UV3f":
            H, W, _ = tex.size()
            tex = tex.flip(0)

        if tex_type == "Constant3f" or tex_type == "UV3f":
            tex = tex.reshape(-1, 3)
        else:
            tex = tex.reshape(-1)

        tex_ = torch2enoki(tex)
        self.append_variable(tex, tex_)

        if tex_type == "Constant1f":
            self.texture_ = pypsdr.ConstantTexture1f(tex_)
        elif tex_type == "Constant3f":
            self.texture_ = pypsdr.ConstantTexture3f(tex_)
        elif tex_type == "UV1f":
            self.texture_ = pypsdr.UVTexture1f(H, W, tex_)
        elif tex_type == "UV3f":
            self.texture_ = pypsdr.UVTexture3f(H, W, tex_)
        else:
            pass


class Material(VarContex):
    def __init__(self):
        """
        This is the base class of different material types.
        """
        super().__init__()


class DiffuseMaterial(Material):
    def __init__(self, tex: Union[FloatTensor, Texture]):
        """
        This material only contains diffuse reflectance ratio of RGB.

        Args:
            tex (Union[FloatTensor, Texture]): diffuse texture data. This can be a tensor of shape (3,) or (H, W, 3)
            for constant texture or texture map
        """
        super().__init__()
        if isinstance(tex, Tensor):
            tex = Texture(tex)
        assert tex.tex_type == "Constant3f" or tex.tex_type == "UV3f"
        self.material_ = pypsdr.DiffuseMaterial(tex.texture_)

        self.merge_variables(*tex.get_variables())


class DiffuseBsdfMaterial(Material):
    def __init__(
        self,
        color: Union[FloatTensor, Texture],
        roughness: Union[FloatTensor, Texture],
        normal: Union[FloatTensor, Texture, None] = None,
    ):
        """Lambertian and Oren-Nayar diffuse reflection

        Args:
            color (Union[FloatTensor, Texture]): Color of surface. It can be a tensor of shape (3,) or (H, W, 3), or a 3-channel Texture instance.
            roughness (Union[FloatTensor, Texture]): Surface roughness. 0 gives standard Lambertian reflection, higher values activate the Oren-Nayar BSDF. It can be a tensor of shape (1,) or (H, W), or a 1-channel Texture instance
            normal (Union[FloatTensor, Texture, None], optional): Normal used for shading. Defaults to None and using geometry normal. It can be a tensor of shape (3,) or (H,W,3), or a 3-channel Texture instance.
        """
        super().__init__()
        if isinstance(color, Tensor):
            color = Texture(color)
        if isinstance(roughness, Tensor):
            roughness = Texture(roughness)
        if isinstance(normal, Tensor):
            normal = Texture(normal)

        assert color.tex_type == "Constant3f" or color.tex_type == "UV3f"
        assert roughness.tex_type == "Constant1f" or roughness.tex_type == "UV1f"
        assert (
            normal is None
            or normal.tex_type == "Constant3f"
            or normal.tex_type == "UV3f"
        )
        self.material_ = pypsdr.DiffuseBsdfMaterial(
            color.texture_,
            roughness.texture_,
            None if normal is None else normal.texture_,
        )
        self.merge_variables(*color.get_variables())
        self.merge_variables(*roughness.get_variables())
        if normal is not None:
            self.merge_variables(*normal.get_variables())


class PrincipledMaterial(Material):
    def __init__(self):
        super().__init__()


class Mesh(VarContex):
    def __init__(
        self,
        vertices: FloatTensor,
        uvs: Union[None, FloatTensor],
        face_indices: IntTensor,
        uv_indices: Union[None, IntTensor],
        material: Material,
    ):
        """Geometry triangle mesh to store an object shape and its material

        Args:
            vertices (FloatTensor): Geometry shape with V vertices. It can be a tensor of shape (V, 3)
            uvs (Union[None, FloatTensor]): UV map coordinates. It can be a tensor of shape (T,2). If constant material textures are used, it can be None.
            face_indices (IntTensor): Face indices of F triangles. It can be an integer tensor of shape (F,3)
            uv_indices (Union[None, IntTensor]): UV coordinate indices of each triangle faces. It can be an integer tensor of shape (F,3). If constant material textures are used, it can be None.
            material (Material): material of this mesh object.
        """
        super().__init__()
        assert vertices.dim() == 2 and vertices.size(-1) == 3
        assert uvs is None or (uvs.dim() == 2 and uvs.size(-1) == 2)
        assert face_indices.dim() == 2 and face_indices.size(-1) == 3
        assert uv_indices is None or (
            uv_indices.dim() == 2 and uv_indices.size(-1) == 3
        )
        vertices_ = torch2enoki(vertices)
        face_indices_ = Vec3iC(face_indices)
        no_uvs = uvs is None or uv_indices is None
        uvs_ = Vec2fC() if no_uvs else Vec2fC(uvs)
        uv_indices_ = Vec3iC() if no_uvs else Vec3iC(uv_indices)
        material_ = material.material_

        self.append_variable(vertices, vertices_)
        self.merge_variables(*material.get_variables())
        self.mesh_ = pypsdr.TriangleMesh(
            vertices_, uvs_, face_indices_, uv_indices_, material_
        )


class Light(VarContex):
    def __init__(self):
        """Base class of different types of lights"""
        super().__init__()


class AreaLight(Light):
    def __init__(
        self,
        vertices: FloatTensor,
        face_indices: IntTensor,
        emit: FloatTensor,
    ):
        """Triangle mesh lights.

        Args:
            vertices (FloatTensor): Light shape with V vertices. It can be a tensor of shape (V, 3)
            face_indices (IntTensor): Face indices of F triangles. It can be an integer tensor of shape (F,3)
            emit (FloatTensor): Light radiance. It can be a tensor of shape (3,)
        """
        super().__init__()
        assert vertices.dim() == 2 and vertices.size(-1) == 3
        assert face_indices.dim() == 2 and face_indices.size(-1) == 3
        assert emit.dim() == 1 and emit.size(0) == 3
        emit = emit.unsqueeze(0)
        vertices_ = torch2enoki(vertices)
        face_indices_ = Vec3iC(face_indices)
        emit_ = torch2enoki(emit)

        self.append_variable(vertices, vertices_)
        self.append_variable(emit, emit_)

        self.light_ = pypsdr.AreaLight(vertices_, face_indices_, emit_)


class EnvLight(Light):
    def __init__(
        self,
        cube_map: Texture,
    ):
        """Cube map based environment light

        Args:
            cube_map (Texture): Light radiance map of six faces of cube.
        """
        super().__init__()
        assert cube_map.tex_type == "UV3f"
        self.merge_variables(*cube_map.get_variables())
        self.light_ = pypsdr.EnvLight(cube_map.texture_)


class SH9Light(Light):
    def __init__(
        self,
        sh9_coeff: FloatTensor,
    ):
        """Second order spherical harmonics lights

        Args:
            sh9_coeff (FloatTensor): 9 coefficients of 3-channel spherical harmonics lights. It can be a tensor of shape (9,3)
        """
        super().__init__()
        assert sh9_coeff.size() == (9, 3)
        sh9_coeff_ = torch2enoki(sh9_coeff)
        self.append_variable(sh9_coeff, sh9_coeff_)
        self.light_ = pypsdr.SH9Light(sh9_coeff_)


class Scene(VarContex):
    def __init__(
        self,
        cameras: List[BaseCamera],
        meshes: List[Mesh],
        lights: List[Light],
    ):
        """Container of every cameras, objects and lights

        Args:
            cameras (List[BaseCamera]): Cameras in a scene, it will render the same count of images as cameras.
            meshes (List[Mesh]): Mesh based objects in a scene
            lights (List[Light]): Lights in a scene
        """
        super().__init__()
        cameras_ = [c.camera_ for c in cameras]
        meshes_ = [m.mesh_ for m in meshes]
        lights_ = [l.light_ for l in lights]

        [self.merge_variables(*c.get_variables()) for c in cameras]
        [self.merge_variables(*m.get_variables()) for m in meshes]
        [self.merge_variables(*l.get_variables()) for l in lights]

        self.scene_ = pypsdr.Scene(cameras_, meshes_, lights_)
        self.cameras = cameras


class PathIntegrator:
    def __init__(
        self,
        enable_light_visable: bool = False,
        n_pass: int = 1,
        spp_interior: int = 1,
        max_bounce: int = 3,
        mis_light_samples: int = 1,
        mis_bsdf_samples: int = 1,
        spp_primary_edge: int = 1,
        spp_secondary_edge: int = 1,
        primary_edge_preprocess_rounds: int = 1,
        secondary_edge_preprocess_rounds: int = 1,
        primary_edge_hypercube_resolution: int = 10000,
        secondary_edge_hypercube_resolution: List[int] = [10000, 6, 6],
    ):
        """Path space differentiable integrator

        Args:
            enable_light_visable (bool, optional): enable lights visible directly by cameras. Defaults to False.
            n_pass (int, optional): Render one or more times of a scene. Defaults to 1.
            spp_interior (int, optional): Samples per pixel of interior integrator. Defaults to 1.
            max_bounce (int, optional): Max bounce times of rays. Defaults to 3.
            mis_light_samples (int, optional): Multiple importance samples count of lights. Defaults to 1.
            mis_bsdf_samples (int, optional): Multiple importance samples count of BSDF. Defaults to 1.
            spp_primary_edge (int, optional): Samples per pixel of primary edge integrator. Only used in differentiable rendering mode. Defaults to 1.
            spp_secondary_edge (int, optional): Samples per pixel of secondary edge integrator. Only used in differentiable rendering mode. Defaults to 1.
            primary_edge_preprocess_rounds (int, optional): Primary edge preprocess rounds. Only used in differentiable rendering mode. Defaults to 1.
            secondary_edge_preprocess_rounds (int, optional): Secondary edge preprocess rounds. Only used in differentiable rendering mode. Defaults to 1.
            primary_edge_hypercube_resolution (int, optional): Hypercube sampler resolution of primary edge preprocess. Only used in differentiable rendering mode. Defaults to 10000.
            secondary_edge_hypercube_resolution (List[int], optional): Hypercube sampler resolution of secondary edge preprocess. Only used in differentiable rendering mode. Defaults to [10000, 6, 6].
        """
        assert n_pass > 0
        assert spp_interior > 0
        assert spp_primary_edge > 0
        assert spp_secondary_edge > 0
        assert max_bounce > 0
        assert mis_light_samples >= 0
        assert mis_bsdf_samples >= 0
        assert primary_edge_preprocess_rounds > 0
        assert secondary_edge_preprocess_rounds > 0
        assert primary_edge_hypercube_resolution > 0
        assert len(secondary_edge_hypercube_resolution) == 3
        for reso in secondary_edge_hypercube_resolution:
            assert reso > 0

        cfg = pypsdr.PathIntegratorConfig()
        cfg.n_pass = n_pass
        cfg.spp_interior = spp_interior
        cfg.enable_light_visable = enable_light_visable
        cfg.spp_primary_edge = spp_primary_edge
        cfg.spp_secondary_edge = spp_secondary_edge
        cfg.max_bounce = max_bounce
        cfg.mis_light_samples = mis_light_samples
        cfg.mis_bsdf_samples = mis_bsdf_samples
        cfg.primary_edge_preprocess_rounds = primary_edge_preprocess_rounds
        cfg.secondary_edge_preprocess_rounds = secondary_edge_preprocess_rounds
        cfg.primary_edge_hypercube_resolution = primary_edge_hypercube_resolution
        cfg.secondary_edge_hypercube_resolution = secondary_edge_hypercube_resolution
        self.integrator_ = pypsdr.PathIntegrator(cfg)

    def renderC(self, scene: Scene) -> List[FloatTensor]:
        imgs_ = self.integrator_.renderC(scene.scene_)
        imgs = []
        for i in range(len(imgs_)):
            H, W = scene.cameras[i].height, scene.cameras[i].width
            img = imgs_[i].torch().reshape(H, W, 3)
            imgs.append(img)
        return imgs

    def renderD(self, scene: Scene) -> List[FloatTensor]:
        enoki_vars, torch_vars = scene.get_variables()
        params = [self.integrator_, scene.scene_] + enoki_vars + torch_vars
        imgs_data = Wrap2torch.apply(*params)
        imgs = []
        for i in range(len(imgs_data)):
            H, W = scene.cameras[i].height, scene.cameras[i].width
            img = imgs_data[i].reshape(H, W, 3)
            imgs.append(img)
        return imgs
