import torch_psdr as dr
import torch
from pywavefront_uv import Wavefront
import imageio
import matplotlib.pyplot as plt
from utilities import translate, scale, rotate

device = "cuda:0"

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
  vertices = translate(scale(vertices, 100), 300)
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
    roughness = torch.tensor([1], dtype=torch.float32, device=device)
    material = dr.DiffuseBsdfMaterial(diffuse_tex, roughness, normal)
    obj = dr.Mesh(vertices, uvs, face_indices, uv_indices, material)
  objs[name] = obj
  return objs

if __name__ == "__main__":
  obj_path = "data/input/cornell_box.obj"
  objs = load_obj(obj_path)
  obj_path = "data/input/spot/spot_triangulated.obj"
  randuvw = torch.rand(3, dtype=torch.float32, device=device) * torch.tensor([2*torch.pi, torch.pi, 2*torch.pi], dtype=torch.float32, device=device)
  pitch, yaw, roll = randuvw[0], randuvw[1], randuvw[2]
  spot_obj = load_spot(obj_path, rotate_angle=(pitch, yaw, roll))
  objs.update(spot_obj)

  # set camera
  look_from = torch.tensor([278, 278, -800], dtype=torch.float32, device=device)
  look_at = torch.tensor([278, 278, 0], dtype=torch.float32, device=device)
  up = torch.tensor([0, 1, 0], dtype=torch.float32, device=device)
  vfov = torch.tensor([torch.pi/180 * 38], dtype=torch.float32, device=device)
  height, width = 600, 600
  camera = dr.Camera(
    look_from = look_from,
    look_at = look_at,
    up = up,
    vfov = vfov,
    height = height,
    width = width,
  )

  # scene
  cameras = [camera]
  lights = [objs["light"]]
  meshes = [obj for name, obj in objs.items() if name != "light"]
  scene = dr.Scene(cameras, meshes, lights)

  integrator = dr.PathIntegrator(
    n_pass = 1,
    spp_interior = 8,
    enable_light_visable=False,
    max_bounce=3,
    mis_light_samples=3,
    mis_bsdf_samples=3,
  )

  imgs = integrator.renderC(scene)
  img = imgs[0].detach().cpu().numpy()
  imageio.imwrite(f"data/output/spot_{pitch}_{yaw}_{roll}.png", img)
  # plt.imshow(img)