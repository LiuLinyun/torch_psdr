import torch
import math
import numpy as np
import os
import imageio


def translate(v, t):
    return v + t


def scale(v, s):
    return v * s


def deg2rad(deg):
    return torch.pi * deg / 180


def rotate(v, pitch, yaw, roll, center=0, is_degree=False):
    if is_degree:
        pitch, yaw, roll = deg2rad(pitch), deg2rad(yaw), deg2rad(roll)
    device = v.device
    dtype = v.dtype
    v -= center
    
    sin_alpha, cos_alpha = torch.sin(pitch), torch.cos(pitch)
    sin_beta, cos_beta = torch.sin(yaw), torch.cos(yaw)
    sin_gamma, cos_gamma = torch.sin(roll), torch.cos(roll)

    R00 = cos_beta * cos_gamma
    R01 = -cos_beta * sin_gamma
    R02 = sin_beta
    R10 = sin_alpha * sin_beta * cos_gamma + cos_alpha * sin_gamma
    R11 = -sin_alpha * sin_beta * sin_gamma + cos_alpha * cos_gamma
    R12 = -sin_alpha * cos_beta
    R20 = -cos_alpha * sin_beta * cos_gamma + sin_alpha * sin_gamma
    R21 = cos_alpha * sin_beta * sin_gamma + sin_alpha * cos_gamma
    R22 = cos_alpha * cos_beta
    R = torch.stack([R00, R01, R02, R10, R11, R12, R20, R21, R22]).reshape(3, 3)
    v = v.matmul(R.transpose(1, 0)) + center
    return v

def get_vertices_neighbors(vertices_cnt, face_indices):
    neighbors = [set() for _ in range(vertices_cnt)]
    max_neighbors_cnt = 0
    faces_cnt = face_indices.shape[0]
    for idx in range(faces_cnt):
        i, j, k = tuple(face_indices[idx].tolist())
        neighbors[i].add(j)
        neighbors[i].add(k)
        neighbors[j].add(i)
        neighbors[j].add(k)
        neighbors[k].add(i)
        neighbors[k].add(j)
        max_neighbors_cnt = max(max_neighbors_cnt, len(neighbors[i]), len(neighbors[j]), len(neighbors[k]))
    nbs = [list(nei) for nei in neighbors]
    return nbs

def laplacian(verts: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    """
    Computes the laplacian matrix.
    The definition of the laplacian is
    L[i, j] =    -1       , if i == j
    L[i, j] = 1 / deg(i)  , if (i, j) is an edge
    L[i, j] =    0        , otherwise
    where deg(i) is the degree of the i-th vertex in the graph.

    Args:
        verts: tensor of shape (V, 3) containing the vertices of the graph
        edges: tensor of shape (E, 2) containing the vertex indices of each edge
    Returns:
        L: Sparse FloatTensor of shape (V, V)
    """
    V = verts.shape[0]

    e0, e1 = edges.unbind(1)

    idx01 = torch.stack([e0, e1], dim=1)  # (E, 2)
    idx10 = torch.stack([e1, e0], dim=1)  # (E, 2)
    idx = torch.cat([idx01, idx10], dim=0).t()  # (2, 2*E)

    # First, we construct the adjacency matrix,
    # i.e. A[i, j] = 1 if (i,j) is an edge, or
    # A[e0, e1] = 1 &  A[e1, e0] = 1
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=verts.device)
    # pyre-fixme[16]: Module `sparse` has no attribute `FloatTensor`.
    A = torch.sparse.FloatTensor(idx, ones, (V, V))

    # the sum of i-th row of A gives the degree of the i-th vertex
    deg = torch.sparse.sum(A, dim=1).to_dense()

    # We construct the Laplacian matrix by adding the non diagonal values
    # i.e. L[i, j] = 1 ./ deg(i) if (i, j) is an edge
    deg0 = deg[e0]
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    deg0 = torch.where(deg0 > 0.0, 1.0 / deg0, deg0)
    deg1 = deg[e1]
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    deg1 = torch.where(deg1 > 0.0, 1.0 / deg1, deg1)
    val = torch.cat([deg0, deg1])
    # pyre-fixme[16]: Module `sparse` has no attribute `FloatTensor`.
    L = torch.sparse.FloatTensor(idx, val, (V, V))

    # Then we add the diagonal values L[i, i] = -1.
    idx = torch.arange(V, device=verts.device)
    idx = torch.stack([idx, idx], dim=0)
    ones = torch.ones(idx.shape[1], dtype=torch.float32, device=verts.device)
    # pyre-fixme[16]: Module `sparse` has no attribute `FloatTensor`.
    L -= torch.sparse.FloatTensor(idx, ones, (V, V))

    return L

def neighbors_mean(vertices, neighbor_indices):
    v = torch.zeros_like(vertices)
    for i in range(len(neighbor_indices)):
        v[i] = torch.mean(vertices[neighbor_indices[i]], dim=0)
    return v

def mesh_edge_loss(vertices, triangle_indices, target_len=0):
    e01 = triangle_indices[:, [0,1]]
    e12 = triangle_indices[:, [1,2]]
    e20 = triangle_indices[:, [2,0]]
    edge_indices = torch.cat([e01, e12, e20], dim=0)
    edge_indices, _ = edge_indices.sort(dim=1)
    out = torch.unique(edge_indices, dim=0).to(dtype=torch.int64)
    edge_len = torch.norm(vertices[out[:,0]] - vertices[out[:,1]], dim=1)
    var = edge_len.var()
    loss = (edge_len.mean() - target_len) ** 2
    return loss, var


def mesh_laplacian_smoothing(vertices, triangle_indices, method="uniform"):
    pass

def mesh_normal_consistency(vertices, triangle_indices):
    triangle_indices = triangle_indices.to(dtype=torch.int64)
    e01 = triangle_indices[:, [0,1]]
    e12 = triangle_indices[:, [1,2]]
    e20 = triangle_indices[:, [2,0]]
    p0 = triangle_indices[:,0]
    p1 = triangle_indices[:,1]
    p2 = triangle_indices[:,2]
    edge_indices = torch.cat([e01, e12, e20], dim=0)
    p = torch.cat([p2, p0, p1], dim=0)
    edge_indices, sort_indices = edge_indices.sort(dim=1)
    unq_edge_indices, inverse_indices = torch.unique(edge_indices, dim=0, return_inverse=True)
    E = unq_edge_indices.size(0)
    ab = [[] for _ in range(E)]
    np_inverse_indices = inverse_indices.detach().cpu().numpy()
    for i in range(inverse_indices.size(0)):
        idx = np_inverse_indices[i]
        ab[idx].append(i)
    ab = [p for p in ab if len(p) == 2]
    ab_indices = p[ab]
    p_share = vertices[unq_edge_indices]
    v_share = p_share[:,1] - p_share[:,0]
    p_left_right = vertices[ab_indices]
    v_left = p_left_right[:,0] - p_share[:,0]
    v_right = p_left_right[:,1] - p_share[:,0]
    n1 = v_share.cross(v_left)
    n2 = v_share.cross(v_right)
    loss = 1 - torch.cosine_similarity(n1, n2, dim=1).abs()
    return loss.mean()

    
    
if __name__ == '__main__':
    vertices = torch.tensor([
        [0,0,0],
        [1,0,0],
        [0,1,0],
        [0,0,1],
    ], dtype=torch.float32)

    face_indices = torch.tensor([
        [0,1,2],
        [0,2,3],
        [0,3,1],
        [1,2,3],
    ], dtype=torch.int64)

    nbs = get_vertices_neighbors(4, face_indices)
    print(nbs)
    mean = neighbors_mean(vertices, nbs)
    print(mean)

def save_obj(obj_path, vertices, face_indices, obj_name="obj", uvs=None, 
            uv_indices=None, normals=None, normal_indices=None,
            diffuse_tex=None, normal_tex=None
    ):
    N = vertices.shape[0]
    F = face_indices.shape[0]
    T = 0 if uvs is None else uvs.shape[0]
    VN = 0 if normals is None else normals.shape[0]

    
    s = f"mtllib {obj_name}.mtl\n"
    for i in range(N):
        s += f"v {vertices[i][0]} {vertices[i][1]} {vertices[i][2]}\n"
    s += "\n\n"


    for i in range(T):
        s += f"vt {uvs[i][0]} {uvs[i][1]}\n"
    s += "\n\n"


    for i in range(VN):
        s += f"vn {normals[i][0]} {normals[i][1]}\n"
    s += "\n\n"

    s += f"\n\no {obj_name}\n"
    s += f"usemtl {obj_name}_mtl\n"
    for i in range(F):
        vi, vj, vk = face_indices[i][0] + 1, face_indices[i][1] + 1, face_indices[i][2] + 1
        vti, vtj, vtk = ("", "", "") if uvs is None else (uv_indices[i][0] + 1, uv_indices[i][1] + 1, uv_indices[i][2]+1)
        vni, vnj, vnk = ("", "", "") if normals is None else (normal_indices[i][0] + 1, normal_indices[i][1] + 1, normal_indices[i][2]+1)

        if uvs is None and normals is None:
            s += f"f {vi} {vj} {vk}\n"
        elif normals is None:
            s += f"f {vi}/{vti} {vj}/{vtj} {vk}/{vtk}\n"
        else:
            s += f"f {vi}/{vti}/{vni} {vj}/{vtj}/{vnj} {vk}/{vtk}/{vnk}\n"
    s += "\n\n"

    open(obj_path, "w").write(s)

    m = f"""
    newmtl {obj_name}_mtl
    Ka 0 0 0
    Ks 0 0 0
    {f"Kd 1 1 1" if diffuse_tex is None else f"map_Kd {obj_name}_diffuse_tex.png"}
    {f"" if normal_tex is None else f"map_Bump {obj_name}_normal_tex.png"}
    """

    out_dir = os.path.dirname(obj_path)

    open(os.path.join(out_dir, f"{obj_name}.mtl"), "w").write(m)

    if diffuse_tex is not None:
        imageio.imwrite(os.path.join(out_dir, f"{obj_name}_diffuse_tex.png"), diffuse_tex)
    if normal_tex is not None:
        imageio.imwrite(os.path.join(out_dir, f"{obj_name}_normal_tex.png"), normal_tex)

    