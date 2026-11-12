import numpy as np
import torch
from dataclasses import dataclass
from mesh_utils import Mesh, GPURayTracer, GPUSampler


@dataclass
class MeshWrapper:
    mesh: Mesh
    sampler: GPUSampler      # sample(n) -> (points, barycentrics, face_idx)
    ray_tracer: GPURayTracer

    @classmethod
    def from_file(cls, path, n_max_samples, scale):
        mesh = Mesh.from_file(path, build_bvh=True, max_leaf_size=5, smooth_normals=True, scale=scale)
        ray_tracer = GPURayTracer(mesh)
        sampler = GPUSampler(mesh, n_max_samples)
        return cls(mesh, sampler, ray_tracer)


def get_camera_rays(mesh, img_size, device, angle=0.0, distance_scale=1.0):
    """
    angle: rotation around the object in radians (positive = CCW around +Z).
           angle=0 keeps the original camera pose.
    """
    n_pixels = img_size * img_size

    mesh_min, mesh_max = mesh.get_bounds()
    mesh_min = np.stack([mesh_min[0], mesh_min[2], -mesh_max[1]], axis=0)
    mesh_max = np.stack([mesh_max[0], mesh_max[2], -mesh_min[1]], axis=0)

    max_extent = max(mesh_max - mesh_min)
    center = (mesh_max + mesh_min) * 0.5

    # --- original camera offset, but rotated around +Z by `angle`
    base_dx = max_extent * 1.0
    base_dy = -max_extent * 1.5
    r_xy = np.hypot(base_dx, base_dy)
    base_theta = np.arctan2(base_dy, base_dx)
    theta = base_theta + angle

    cam_pos = np.array([
        center[0] + r_xy * np.cos(theta) * distance_scale,
        center[1] + r_xy * np.sin(theta) * distance_scale,
        center[2] + max_extent * 0.5 * distance_scale,   # keep the same "slightly top" height
    ], dtype=np.float32)

    cam_poses = np.tile(cam_pos, (n_pixels, 1))

    # forward vector (keep same scaling as your original code)
    cam_dir = (center - cam_pos) * 0.9 / distance_scale

    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    x_dir = np.cross(cam_dir, up)
    x_dir = x_dir / np.linalg.norm(x_dir) * (max_extent / 2)

    y_dir = -np.cross(x_dir, cam_dir)
    y_dir = y_dir / np.linalg.norm(y_dir) * (max_extent / 2)

    x_coords, y_coords = np.meshgrid(
        np.linspace(-1, 1, img_size),
        np.linspace(-1, 1, img_size),
    )
    x_coords = x_coords.flatten()
    y_coords = y_coords.flatten()

    dirs = cam_dir[None, :] + x_dir[None, :] * x_coords[:, None] + y_dir[None, :] * y_coords[:, None]

    d_cam_poses = torch.from_numpy(cam_poses).float().to(device)
    d_dirs = torch.from_numpy(dirs).float().to(device)

    d_cam_poses = torch.stack([d_cam_poses[:, 0], d_cam_poses[:, 2], -d_cam_poses[:, 1]], dim=-1)
    d_dirs = torch.stack([d_dirs[:, 0], d_dirs[:, 2], -d_dirs[:, 1]], dim=-1)

    return d_cam_poses, d_dirs


def sample_uniform_directions(n, device):
    """Unit directions distributed uniformly over the sphere."""
    ds = torch.randn((n, 3), device=device)
    return ds / ds.norm(dim=1, keepdim=True)
