import os

# import torch_psdr as dr
import src.python.torch_psdr as dr
import torch
import torch.optim
import torch.nn.functional as F

# import torch.cuda
from pywavefront_uv import Wavefront
import random
import imageio.v2 as imageio
from utilities import *
import numpy as np
import gc
import datetime
import time

print("imported all!")

device = "cuda:0"
global_scale = 130
global_translation = torch.tensor([300, 180, 300], dtype=torch.float32, device=device)


def load_unit_sphere_mesh(sphere_obj_path):
    # return vertices, uvs, triangle_indices, uv_indices
    scene = Wavefront(sphere_obj_path, create_materials=True, collect_faces=True)
    vertices = np.array(scene.vertices)
    uvs = np.array(scene.parser.tex_coords)
    obj = scene.meshes["sphere"]
    indices = np.array(obj.faces, dtype=np.int32)
    triangle_indices = indices[:, :, 0]
    uv_indices = triangle_indices.copy()
    return vertices, uvs, triangle_indices, uv_indices


def load_obj(obj_path):
    scene = Wavefront(obj_path, create_materials=True, collect_faces=True)
    vertices = torch.tensor(scene.vertices, dtype=torch.float32, device=device)
    uvs = (
        None
        if scene.parser.tex_coords == []
        else torch.tensor(scene.parser.tex_coords, dtype=torch.float32, device=device)
    )
    objs = {}
    for name, mesh in scene.meshes.items():
        if name is None:
            name = str(hash(mesh))
        indices = torch.tensor(mesh.faces, dtype=torch.int32, device=device)
        face_indices = indices[:, :, 0]
        uv_indices = indices[:, :, 1] if indices[0, 0, 1] != -1 else None
        material = mesh.materials[0]
        obj = None
        if name == "light":
            emit = torch.tensor(
                material.ambient[:3], dtype=torch.float32, device=device
            )
            obj = dr.AreaLight(vertices, face_indices, emit)
        elif name == "short_block":
            continue
        else:
            diffuse = torch.tensor(
                material.diffuse[:3], dtype=torch.float32, device=device
            )
            normal = None
            roughness = torch.tensor([0.5], dtype=torch.float32, device=device)
            material = dr.DiffuseBsdfMaterial(diffuse, roughness, normal)
            obj = dr.Mesh(vertices, uvs, face_indices, uv_indices, material)
        objs[name] = obj
    return objs


def load_spot(obj_path, scale=None, rotate_angle=None, translation=None):
    scene = Wavefront(obj_path, create_materials=True, collect_faces=True)
    vertices = torch.tensor(scene.vertices, dtype=torch.float32, device=device)
    if scale is not None:
        vertices.mul_(scale)
    if rotate_angle is not None:
        pitch, yaw, roll = rotate_angle
        vertices = rotate(vertices, pitch, yaw, roll)
    if translation is not None:
        vertices.add_(translation)
    uvs = (
        None
        if scene.parser.tex_coords == []
        else torch.tensor(scene.parser.tex_coords, dtype=torch.float32, device=device)
    )
    objs = {}
    for name, mesh in scene.meshes.items():
        if name is None:
            name = str(hash(mesh))
        indices = torch.tensor(mesh.faces, dtype=torch.int32, device=device)
        face_indices = indices[:, :, 0]
        uv_indices = indices[:, :, 1] if indices[0, 0, 1] != -1 else None
        material = mesh.materials[0]
        tex = imageio.imread(material.texture.path)
        diffuse_tex = torch.tensor(tex, dtype=torch.float32, device=device) / 255
        normal = None
        roughness = torch.tensor([0.5], dtype=torch.float32, device=device)
        material = dr.DiffuseBsdfMaterial(diffuse_tex, roughness, normal)
        obj = dr.Mesh(vertices, uvs, face_indices, uv_indices, material)
    objs[name] = obj
    return objs


def set_camera():
    look_from = torch.tensor([278, 150, -600], dtype=torch.float32, device=device)
    look_at = torch.tensor([278, 150, 0], dtype=torch.float32, device=device)
    up = torch.tensor([0, 1, 0], dtype=torch.float32, device=device)
    vfov = torch.tensor(
        [torch.deg2rad(torch.tensor(30.0))], dtype=torch.float32, device=device
    )
    height, width = 600, 600
    camera = dr.PerspectiveCamera(
        look_from=look_from,
        look_at=look_at,
        up=up,
        vfov=vfov,
        height=height,
        width=width,
    )
    # camera = dr.OrthographicCamera(
    #     look_from=look_from,
    #     look_at=look_at,
    #     up=up,
    #     view_width=torch.tensor([900,], dtype=torch.float32, device=device),
    #     height=height,
    #     width=width
    # )
    return camera


def render_img(scene, diff_mode=False, spp=None):
    spp = 11 if spp is None else spp
    integrator = dr.PathIntegrator(
        n_pass=1,
        spp_interior=spp,
        enable_light_visable=False,
        spp_primary_edge=2,
        spp_secondary_edge=2,
        primary_edge_preprocess_rounds=4,
        secondary_edge_preprocess_rounds=4,
        max_bounce=2,
        mis_light_samples=1,
        mis_bsdf_samples=0,
    )
    imgs = integrator.renderD(scene) if diff_mode else integrator.renderC(scene)
    return imgs


def render_tgt(
    obj_path,
    spot_path,
    scale=None,
    rotate_angle=None,
    translation=None,
    diff_mode=False,
    spp=None,
):
    objs = load_obj(obj_path)
    spot = load_spot(spot_path, scale, rotate_angle, translation)
    objs.update(spot)
    lights = [obj for name, obj in objs.items() if name == "light"]
    meshes = [obj for name, obj in objs.items() if name != "light"]
    cameras = [set_camera()]
    scene = dr.Scene(cameras, meshes, lights)
    imgs = render_img(scene, diff_mode, spp)
    return imgs


def render_src(
    obj_path,
    mesh_info,
    scale=None,
    rotate_angle=None,
    translation=None,
    diff_mode=False,
    spp=None,
):
    vertices, uvs, triangle_indices, uv_indices, diffuse_uvmap = mesh_info
    if scale is not None:
        vertices.mul_(scale)
    if rotate_angle is not None:
        pitch, yaw, roll = rotate_angle
        vertices = rotate(vertices, pitch, yaw, roll)
    if translation is not None:
        vertices.add_(translation)
    roughness = torch.tensor([0.5], dtype=torch.float32, device=device)
    normal = None
    material = dr.DiffuseBsdfMaterial(diffuse_uvmap, roughness, normal)
    mesh = dr.Mesh(vertices, uvs, triangle_indices, uv_indices, material)

    objs = load_obj(obj_path)
    lights = [obj for name, obj in objs.items() if name == "light"]
    meshes = [obj for name, obj in objs.items() if name != "light"] + [
        mesh,
    ]
    cameras = [set_camera()]

    scene = dr.Scene(cameras, meshes, lights)
    imgs = render_img(scene, diff_mode, spp)
    return imgs[0]


def render_tgts(obj_path, spot_path, spp=72):
    pitchs = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    yaws = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    rolls = np.linspace(0, np.pi, 8, endpoint=False)
    rst = []
    for pitch in pitchs:
        for yaw in yaws:
            for roll in rolls:
                angle = (pitch, yaw, roll)
                print(f"render with angle: {angle}")
                imgs = render_tgt(
                    obj_path,
                    spot_path,
                    global_scale,
                    angle,
                    global_translation,
                    diff_mode=False,
                    spp=spp,
                )
                imageio.imwrite(
                    os.path.join("/workspace/mnt/enoki-mask/data/output/tgt_imgs", f"tgt_{angle}.png"),
                    torch.clamp(imgs[0],0,1).detach().cpu().numpy(),
                )
                rst.append((angle, imgs[0]))
    return rst


def main():
    import enoki as ek

    obj_path = "/workspace/mnt/enoki-mask/data/input/cornell_box.obj"
    spot_path = "/workspace/mnt/enoki-mask/data/input/spot/spot_triangulated.obj"
    sphere_path = "/workspace/mnt/enoki-mask/data/input/sphere/sphere.obj"
    tgt_pkl_path = "/workspace/mnt/enoki-mask/data/output/multi_view_bigger.pkl"
    output_dir = "/workspace/mnt/enoki-mask/data/output/opt_pts/"
    param_pkl_path = "/workspace/mnt/enoki-mask/data/output/opt_pts/param_epoch_2_2022-12-23 12:55:59.978722.pkl"
    if os.path.exists(tgt_pkl_path):
        tgts = torch.load(tgt_pkl_path, map_location="cpu")
        # print(tgts)
    else:
        tgts = render_tgts(obj_path, spot_path, spp=90)
        torch.save(tgts, tgt_pkl_path)

    # for angle, tgt in tgts:
    #   img = tgt.detach().cpu().numpy()
    #   imageio.imwrite(f"/workspace/mnt/enoki-mask/data/output/tgt_imgs/tgt_{angle}.png", img)
    # return

    vertices, uvs, triangle_indices, uv_indices = load_unit_sphere_mesh(sphere_path)
    uvs = torch.tensor(uvs, dtype=torch.float32, device=device)
    triangle_indices = torch.tensor(triangle_indices, dtype=torch.int32, device=device)
    uv_indices = torch.tensor(uv_indices, dtype=torch.int32, device=device)

    # scale_param = torch.tensor(150, dtype=torch.float32, device=device, requires_grad=True)
    if os.path.exists(param_pkl_path):
        param_dict = torch.load(param_pkl_path)
        vertices_param = param_dict["vertices"]
        tex_param = param_dict["tex"]
        # tex_param = torch.zeros(
        #     (512, 512, 3), dtype=torch.float32, device=device, requires_grad=True
        # )
        # tex = torch.sigmoid(tex_param).detach().cpu().numpy()
        # imageio.imwrite(os.path.join(output_dir, "tex.png"), tex)

        print(param_dict)
    else:
        vertices_param = torch.tensor(
            vertices, dtype=torch.float32, device=device, requires_grad=True
        )
        tex_param = torch.zeros(
            (512, 512, 3), dtype=torch.float32, device=device, requires_grad=True
        )

        # tex_param_ = imageio.imread("/workspace/mnt/enoki-mask/data/input/spot/spot_texture.png")
        # tex_param = torch.tensor(tex_param_, dtype=torch.float32, device=device, requires_grad=True)

    if False:
        v = vertices_param.detach().cpu().numpy()
        vt = uvs.detach().cpu().numpy()
        f = triangle_indices.detach().cpu().numpy()
        vt_idx = uv_indices.detach().cpu().numpy()
        diffuse = torch.sigmoid(tex_param).detach().cpu().numpy()
        save_obj(os.path.join(output_dir, "../optimized_obj/optimized_spot.obj"), v, f, obj_name="optimized_spot",
            uvs=vt, uv_indices=vt_idx, diffuse_tex=diffuse
        )
        return

    # >>>
    neighbor_indices = get_vertices_neighbors(vertices.shape[0], triangle_indices)
    # neighbors_mean(vertices_param, neighbor_indices)
    # return
    # <<<

    lr = 1
    parameters = [
        {"params": vertices_param, "lr": lr * 1e-8}, # 1e-8
        {"params": tex_param, "lr": lr * 1e6},
    ]
    optim = torch.optim.SGD(parameters, lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, 40, 0.8)
    max_epoch = 600
    for epoch in range(max_epoch):
        batch=16
        random.shuffle(tgts)
        tgts_grouped = [tgts[i:i+batch] for i in range(0, len(tgts), batch)]
        for batch_idx, curr_tgts in enumerate(tgts_grouped):
            for idx, (rotate_angle, tgt_img) in enumerate(curr_tgts):
                # tgt_idx = random.randint(0, len(tgts) - 1)
                # rotate_angle, tgt_img = tgts[tgt_idx]
                tgt_img = tgt_img.to(device=device)
                # tgt_img = tgt_img.permute(2,0,1).unsqueeze(0)
                # tgt_img = F.interpolate(tgt_img, size=(128,128), mode="bilinear")
                # tgt_img = tgt_img.squeeze().permute(1,2,0)
                # np_tgt_img = tgt_img.detach().cpu().numpy()
                # imageio.imwrite(os.path.join(output_dir, f"atgt{idx}.png"), np_tgt_img)
                # vertices = torch.sigmoid(vertices_param)
                vertices = vertices_param * 130
                # vertices = F.hardtanh(vertices_param, -400, 400)
                tex = torch.sigmoid(tex_param)
                mesh_info = (vertices, uvs, triangle_indices, uv_indices, tex)

                t1 = time.perf_counter()
                img = render_src(
                    obj_path,
                    mesh_info,
                    None,
                    rotate_angle,
                    global_translation,
                    diff_mode=True,
                    spp=None,
                )
                t2 = time.perf_counter()

                if epoch == 0 and batch_idx == 0 and idx == 0:
                    imageio.imwrite(
                        os.path.join(output_dir, f"epoch_start.png"),
                        img.detach().cpu().numpy(),
                    )

                loss_img = F.mse_loss(img, tgt_img)  # img loss
                # loss_smooth = F.mse_loss(vertices, vertices_neighbors_mean)
                # loss_edge_reg, loss_edge_var = mesh_edge_loss(vertices, triangle_indices)
                # loss_normal = mesh_normal_consistency(vertices, triangle_indices)
                # loss = (
                #     loss_img
                #     + loss_smooth * 10
                #     + loss_edge_reg * 10
                #     + loss_edge_var * 100
                #     + loss_normal * 100
                # )
                loss = loss_img / batch
                if idx + 1 == len(curr_tgts):
                    vertices_neighbors_mean = neighbors_mean(vertices_param.detach(), neighbor_indices)
                    loss_smooth = F.mse_loss(vertices_param, vertices_neighbors_mean)
                    loss_edge_reg, loss_edge_var = mesh_edge_loss(vertices_param, triangle_indices)
                    # loss_normal = mesh_normal_consistency(vertices_param, triangle_indices)
                    loss += (
                        loss_smooth * 8e10
                        + loss_edge_reg * 8e9
                        # + loss_edge_var * 1e0
                        # + loss_normal * 1e0
                    )
                    print(
                        f"epoch: {epoch}, batch_idx: {batch_idx}, loss: {loss:.4}, loss_smooth: {loss_smooth:.4}, vert grad max: {torch.max(vertices_param.grad):.4}")
                    # loss_edge_reg: {loss_edge_reg:.4}, loss_edge_var: {loss_edge_var:.4}, loss_normal: {loss_normal:.4},\
                loss.backward()
                # print(
                #     f"iter: {iter}, loss: {loss:.4}, loss_img: {loss_img:.4}, loss_smooth: {loss_smooth:.4},\
                #     loss_edge_reg: {loss_edge_reg:.4}, loss_edge_var: {loss_edge_var:.4},\
                #     loss_normal: {loss_normal:.4},"
                # )
                print(
                    f"epoch: {epoch}, batch_idx: {batch_idx}, idx: {idx}, used {(t2-t1)*1000}ms, loss_img: {loss_img:.4}, grad mean: vertices: {torch.mean(vertices_param.grad):.4}, tex {torch.mean(tex_param.grad):.4}, vertices grad max: {torch.max(vertices_param.grad):.4}"
                )
            optim.step()
            optim.zero_grad()
            imageio.imwrite(
                os.path.join(output_dir, f"epoch_{epoch:04}_batch_{batch_idx:04}.png"),
                torch.clamp(img, 0, 1).detach().cpu().numpy(),
            )
        scheduler.step()
        tm = datetime.datetime.now()
        torch.save(
            {"vertices": vertices_param, "tex": tex_param},
            os.path.join(output_dir, f"param_epoch_{epoch}_{tm}.pkl"),
        )


if __name__ == "__main__":
    main()
