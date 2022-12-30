# import torch_psdr as dr
import src.python.torch_psdr as dr
import torch
import torch.optim
import torch.nn.functional as F
# import torch.cuda
from pywavefront_uv import Wavefront
import random
import imageio.v2 as imageio
from utilities import translate, scale, rotate
import numpy as np

import enoki as ek

# define some global variables
device = "cuda"
tex_H, tex_W = 1024, 1024

def load_obj(obj_path):
  scene = Wavefront(obj_path, create_materials=True, collect_faces=True)
  vertices = torch.tensor(scene.vertices, dtype=torch.float32, device=device)
  uvs = None if scene.parser.tex_coords == [] else torch.tensor(scene.parser.tex_coords, dtype=torch.float32, device=device)
  objs = {}
  for name, mesh in scene.meshes.items():
    if name is None:
      name = str(hash(mesh))
    indices = torch.tensor(mesh.faces, dtype=torch.int32, device=device)
    face_indices = indices[:,:,0]
    uv_indices = indices[:,:,1] if indices[0,0,1] != -1 else None
    material = mesh.materials[0]
    obj = None
    if name == "light":
      emit = torch.tensor(material.ambient[:3], dtype=torch.float32, device=device)
      obj = dr.AreaLight(vertices, face_indices, emit)
    else:
      diffuse = torch.tensor(material.diffuse[:3], dtype=torch.float32, device=device)
      normal = None
      roughness = torch.tensor([0.5], dtype=torch.float32, device=device)
      material = dr.DiffuseBsdfMaterial(diffuse, roughness, normal)
      obj = dr.Mesh(vertices, uvs, face_indices, uv_indices, material)
    objs[name] = obj
  return objs

def load_spot(obj_path, diffuse_tex=None, rotate_angle=None):
  scene = Wavefront(obj_path, create_materials=True, collect_faces=True)
  vertices = torch.tensor(scene.vertices, dtype=torch.float32, device=device)
  if rotate_angle is not None:
    pitch, yaw, roll = rotate_angle
    vertices = rotate(vertices, pitch, yaw, roll)
  vertices.mul_(150).add_(300)
  uvs = None if scene.parser.tex_coords == [] else torch.tensor(scene.parser.tex_coords, dtype=torch.float32, device=device)
  objs = {}
  for name, mesh in scene.meshes.items():
    if name is None:
      name = str(hash(mesh))
    indices = torch.tensor(mesh.faces, dtype=torch.int32, device=device)
    face_indices = indices[:,:,0]
    uv_indices = indices[:,:,1] if indices[0,0,1] != -1 else None
    material = mesh.materials[0]
    if diffuse_tex is None:
      tex = imageio.imread(material.texture.path)
      diffuse_tex = torch.tensor(tex, dtype=torch.float32, device=device) / 255
    normal = None
    roughness = torch.tensor([0.5], dtype=torch.float32, device=device)
    material = dr.DiffuseBsdfMaterial(diffuse_tex, roughness, normal)
    obj = dr.Mesh(vertices, uvs, face_indices, uv_indices, material)
  objs[name] = obj
  return objs

def set_camera():
  look_from = torch.tensor([278, 278, -800], dtype=torch.float32, device=device)
  look_at = torch.tensor([278, 278, 0], dtype=torch.float32, device=device)
  up = torch.tensor([0, 1, 0], dtype=torch.float32, device=device)
  vfov = torch.tensor([np.pi * 38 / 180], dtype=torch.float32, device=device)
  height, width = 500, 500
  camera = dr.Camera(
    look_from = look_from,
    look_at = look_at,
    up = up,
    vfov = vfov,
    height = height,
    width = width,
  )
  return camera

def render_img(scene, diff_mode=False):
  integrator = dr.PathIntegrator(
    n_pass=1,
    spp_interior=64,
    enable_light_visable=False,
    spp_primary_edge=1,
    spp_secondary_edge=1,
    primary_edge_preprocess_rounds=1,
    secondary_edge_preprocess_rounds=1,
    max_bounce=3,
    mis_light_samples=3,
    mis_bsdf_samples=3,
  )
  imgs = integrator.renderD(scene) if diff_mode else integrator.renderC(scene)
  return imgs

def render_once(obj_path, spot_path, diffuse_tex=None, spot_rotation=None, diff_mode=False):
  objs = load_obj(obj_path)
  spot = load_spot(spot_path, diffuse_tex, spot_rotation)
  objs.update(spot)

  lights = [obj for name, obj in objs.items() if name == "light"]
  meshes = [obj for name, obj in objs.items() if name != "light"]

  cameras = [set_camera()]

  scene = dr.Scene(cameras, meshes, lights)

  imgs = render_img(scene, diff_mode)

  return imgs

def main():
  obj_path = "/workspace/mnt/enoki-mask/data/input/cornell_box.obj"
  spot_path = "/workspace/mnt/enoki-mask/data/input/spot/spot_triangulated.obj"

  tex_param = torch.zeros((tex_H, tex_W, 3), dtype=torch.float32, device=device, requires_grad=True)
  optim = torch.optim.SGD([tex_param], lr=1)

  iterations = 100
  for iter in range(iterations):
    print(f"iter: {iter}")
    pitch, yaw, roll = random.random() * 2 * np.pi, random.random() * 2 * np.pi, random.random() * np.pi
    target_imgs = render_once(obj_path, spot_path, diffuse_tex=None, spot_rotation=(pitch, yaw, roll), diff_mode=False)
    tex_param_ = torch.sigmoid(tex_param)
    imgs = render_once(obj_path, spot_path, tex_param_, (pitch, yaw, roll), diff_mode=True)
    loss = F.mse_loss(imgs[0], target_imgs[0]) * 1e8
    loss.backward()
    optim.step()

    print(f"iter: {iter}; avg grad: {torch.mean(tex_param.grad)}; loss: {loss}")
    optim.zero_grad()

    if iter%5 == 4:
      tex = torch.sigmoid(tex_param_).detach().cpu().numpy()
      imageio.imwrite(f"/workspace/mnt/enoki-mask/data/output/tex_{iter}.png", tex)
    optim.zero_grad()

if __name__ == "__main__":
  main()
  print("ALL DONE!")


