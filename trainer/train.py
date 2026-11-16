import os
import json
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR
import torch.nn.functional as F
import copy
from tqdm import tqdm
from mesh_utils import trace_shell_segments
from utils import MeshWrapper, sample_uniform_directions, get_camera_rays
from visualization import save_mesh_previews, render_predictions, render_predictions_with_neural_renderer
import sys
from models import RayModel
import time

# to make tensorflow shut up
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from torch.utils.tensorboard import SummaryWriter

eps = 1e-8

# seed everything for reproducibility
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def save_checkpoint(cfg, model_config, model, averaged_model, optimizer, scheduler, run_name, step):
    checkpoint_data = {
        "model_config": model_config,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "learning_rate_scheduler": scheduler.state_dict()
    }

    if cfg.train.use_averaged_model in ['EMA', 'SWA']:
        checkpoint_data["averaged_model"] = averaged_model.state_dict()
        copy_model = copy.deepcopy(averaged_model.module)
    else:
        copy_model = copy.deepcopy(model)

    copy_model.half()

    if cfg.train.save_pt:
        torch.save(copy_model.state_dict(), f"{cfg.train.checkpoints_path}/{run_name}_half.pt")
        torch.save(checkpoint_data, f"{cfg.train.checkpoints_path}/{run_name}.pt")

    if cfg.train.save_bin:
        point_encoding_params = copy_model.point_encoding.params.data.cpu().numpy().astype(np.float16)
        direction_encoding_params = copy_model.direction_encoding.params.data.cpu().numpy().astype(np.float16)
        network_params = copy_model.network.params.data.cpu().numpy().astype(np.float16)
        total_params = np.concatenate([point_encoding_params, direction_encoding_params, network_params])
        total_params.tofile(f"{cfg.train.checkpoints_path}/{run_name}.bin")


@torch.no_grad()
def eval_model(cfg, fine_mesh, outer_mesh, inner_mesh, model):
    model.eval()

    cam_poses, ds = get_camera_rays(fine_mesh.mesh, img_size=cfg.visualization.image_size, device=cfg.device, distance_scale=1.0, angle=cfg.visualization.camera_angle)
    ds = ds / ds.norm(dim=1, keepdim=True)

    x_orig_mask, x_orig_t, x_orig_normals, _, colors, *_ = fine_mesh.ray_tracer.trace(cam_poses, ds, allow_negative=False)
    x_orig = cam_poses + ds * x_orig_t[:, None]
    x_orig[~x_orig_mask] = 0

    pred_intersection_global = torch.zeros((cam_poses.shape[0],), dtype=torch.bool, device=cfg.device)
    pred_t_global = torch.zeros((cam_poses.shape[0],), dtype=torch.float32, device=cfg.device)
    pred_normal_global = torch.zeros((cam_poses.shape[0], 3), dtype=torch.float32, device=cfg.device)
    pred_colors_global = torch.zeros((cam_poses.shape[0], 3), dtype=torch.float32, device=cfg.device)

    x_outer_enter_mask, x_outer_enter_t, *_ = outer_mesh.ray_tracer.trace(
        cam_poses, 
        ds,
        allow_negative=False,
        allow_backward=False,
        allow_forward=True,        
    )
    x_outer_enter = cam_poses + ds * x_outer_enter_t[:, None]
    x_outer_enter = x_outer_enter[x_outer_enter_mask]

    mask_for_remaining_rays_global = x_outer_enter_mask.clone()

    # directions for rays left to trace
    ds_left = ds[x_outer_enter_mask]

    # distance already accumulated along traced segments
    accum_t = x_outer_enter_t[x_outer_enter_mask]

    i = 0
    while len(x_outer_enter) > 0:
        x_outer_enter = x_outer_enter + ds_left * eps

        # intersect orig mesh, inner shell and outer shell again 
        x_outer_exit_mask, x_outer_exit_t, *_ = outer_mesh.ray_tracer.trace(
            x_outer_enter,
            ds_left,
            allow_negative=False,
            allow_backward=True,
            allow_forward=False,            
        )
        x_outer_exit = x_outer_enter + ds_left * x_outer_exit_t[:, None]

        if (~x_outer_exit_mask).sum() > 0:
            x_outer_exit_t[~x_outer_exit_mask] = 1e-8
            x_outer_exit[~x_outer_exit_mask] = x_outer_enter[~x_outer_exit_mask] + ds_left[~x_outer_exit_mask] * x_outer_exit_t[~x_outer_exit_mask][:, None]

        x_inner_mask, x_inner_t, *_ = inner_mesh.ray_tracer.trace(x_outer_enter, ds_left, allow_negative=False)
        x_inner_enter = x_outer_enter + ds_left * x_inner_t[:, None]
        x_inner_enter[~x_inner_mask] = 0


        input_enter_points = x_outer_enter
        input_exit_points = torch.zeros_like(input_enter_points)
        inner_apply = x_inner_mask & (x_inner_t < x_outer_exit_t)
        input_exit_points[inner_apply] = x_inner_enter[inner_apply]
        input_exit_points[~inner_apply] = x_outer_exit[~inner_apply]
        input_directions = ds_left

        pred_intersection, pred_t, pred_normal, pred_colors = model(
            input_enter_points,
            input_exit_points,
            input_directions,
        )
        pred_intersection_mask = (pred_intersection >= 0)
        pred_intersection_mask[x_inner_mask & (x_inner_t < x_outer_exit_t)] = True

        pred_intersection_global[mask_for_remaining_rays_global] = pred_intersection_mask
        pred_t_global[mask_for_remaining_rays_global] = pred_t + accum_t
        pred_normal_global[mask_for_remaining_rays_global] = pred_normal
        pred_colors_global[mask_for_remaining_rays_global] = pred_colors

        x_outer_exit = x_outer_exit + ds_left * eps
        x_outer_enter_mask_new, x_outer_enter_t_new, *_ = outer_mesh.ray_tracer.trace(
            x_outer_exit,
            ds_left,
            allow_negative=False,
            allow_backward=False,
            allow_forward=True,
        )
        x_outer_enter_new = x_outer_exit + ds_left * x_outer_enter_t_new[:, None]        

        # prepare for next iteration
        mask_for_remaining_rays = ~pred_intersection_mask & (x_outer_enter_mask_new | x_inner_mask)
        mask_for_remaining_rays_global[mask_for_remaining_rays_global.clone()] = mask_for_remaining_rays
        x_outer_enter = x_outer_enter_new[mask_for_remaining_rays]
        ds_left = ds_left[mask_for_remaining_rays]
        accum_t = accum_t[mask_for_remaining_rays] + x_outer_exit_t[mask_for_remaining_rays] + x_outer_enter_t_new[mask_for_remaining_rays]


        i += 1
        if i > 10:
            break


    # prepare inputs for rendering

    true_mask = x_orig_mask.clone()
    points = cam_poses[true_mask] + ds[true_mask] * x_orig_t[true_mask][:, None]
    normals_traced = x_orig_normals.clone()

    whole_intersected_mask = pred_intersection_global.clone()

    predicted_points = (cam_poses + ds * pred_t_global[:, None])[whole_intersected_mask]

    whole_predicted_normal = pred_normal_global.clone()

    mse, psnr, accuracy = render_predictions(
        cfg, points, predicted_points, cam_poses, 
        normals_traced, whole_predicted_normal, 
        whole_intersected_mask, true_mask,
        pred_colors_global, colors,
    )
    return mse, psnr, accuracy


def generate_segments(outer_mesh, inner_mesh, fine_mesh, x, ds, max_segments=11, min_rays=10):
    """Training segments for rays (x, ds), walked through the double shell on the GPU.

    One mesh_utils kernel walks every ray through all its segments; this
    function then keeps the same segments, in the same order, as the original
    per-iteration torch loop: iteration-major, ray order preserved within an
    iteration, and stopping before the first iteration with <= min_rays rays.
    """
    seg = trace_shell_segments(
        outer_mesh.ray_tracer, inner_mesh.ray_tracer, fine_mesh.ray_tracer,
        x, ds, max_segments=max_segments, shift_eps=eps, no_exit_t=1e-3,
    )
    n_iters = 0
    for count in seg.valid.sum(dim=1).tolist():  # host sync 1
        if count <= min_rays:
            break
        n_iters += 1

    n_rays = x.shape[0]
    idx = seg.valid[:n_iters].reshape(-1).nonzero().squeeze(1)  # host sync 2
    rows = seg.records[:n_iters].reshape(n_iters * n_rays, -1).index_select(0, idx)

    gt_mask = rows[:, 19] > 0.5
    gt_idx = gt_mask.nonzero().squeeze(1)  # host sync 3: indices of segments with a gt hit

    return (
        gt_mask,            # gt intersection mask
        gt_idx,             # indices where gt_mask is True
        rows[:, 12:15],     # gt normals
        rows[:, 18],        # gt distance
        rows[:, 15:18],     # gt colors
        rows[:, 0:3],       # segment enter points
        rows[:, 3:6],       # segment exit points (inner shell hit or outer exit)
        rows[:, 6:9],       # directions
        rows[:, 9:12],      # outer shell normals at entry
    )


class BatchPrefetcher:
    """Generates training batches on a side CUDA stream.

    The host syncs inside generate_segments() then wait only for that stream,
    so the next batch is produced while the GPU is still running the previous
    model step. Batches (and RNG consumption order) are the same as generating
    them inline at the top of each epoch.
    """

    def __init__(self, cfg, fine_mesh, outer_mesh, inner_mesh):
        self.cfg = cfg
        self.meshes = (outer_mesh, inner_mesh, fine_mesh)
        self.stream = torch.cuda.Stream()

    def prepare(self):
        outer_mesh, inner_mesh, fine_mesh = self.meshes
        with torch.cuda.stream(self.stream):
            x, _, _ = outer_mesh.sampler.sample(self.cfg.train.sample_size)  # (points, barycentrics, face_idx)
            ds = sample_uniform_directions(x.shape[0], self.cfg.device)
            return generate_segments(outer_mesh, inner_mesh, fine_mesh, x, ds)

    def hand_over(self, batch):
        """Make a prepared batch safe to use on the current stream."""
        main = torch.cuda.current_stream()
        main.wait_stream(self.stream)
        for t in batch:
            t.record_stream(main)
        return batch


def write_losses_to_tensorboard(writer, pending, evaluation_interval):
    """Write a deferred epoch's scalars once its loss values reached the host."""
    epoch, event, host_vals, lr, mse, psnr, accuracy, flip, n = pending
    event.synchronize()  # waits for that epoch's step only, not for work queued after it
    total_v, normal_v, distance_v, cls_v, color_v = host_vals.tolist()
    writer.add_scalar("Total_loss", total_v, n)
    writer.add_scalar("Normal_loss", normal_v, n)
    writer.add_scalar("Distance_loss", distance_v, n)
    writer.add_scalar("Entropy_loss", cls_v, n)
    writer.add_scalar("Learning_rate", lr, n)
    writer.add_scalar("Color_loss", color_v, n)
    if mse > 0:
        writer.add_scalar("MSE", mse, n)
    if psnr > 0:
        writer.add_scalar("PSNR", psnr, n)
    if accuracy > 0:
        writer.add_scalar("Accuracy", accuracy, n)
    if flip > 0:
        writer.add_scalar("FLIP", flip, n)
    # SummaryWriter also auto-flushes every 120 s; force it only on evaluation epochs
    if epoch % evaluation_interval == evaluation_interval - 1:
        writer.flush()


def compute_losses(cfg, model, inner_mesh, batch, log_baseline=False):
    """Loss terms for one batch of segments (see generate_segments for the layout).

    Returns (total, normal, distance, cls, color). Distance and color are computed for
    logging only: with the current weights they do not enter the total.
    """
    (gt_mask, gt_idx, gt_normals, gt_distance, gt_colors,
     enter_points, exit_points, directions, outer_normals) = batch

    pred_logit, pred_t, pred_normals, pred_colors = model(enter_points, exit_points, directions)

    color_loss = F.mse_loss(pred_colors.index_select(0, gt_idx), gt_colors.index_select(0, gt_idx))
    distance_loss = F.l1_loss(pred_t.index_select(0, gt_idx), gt_distance.index_select(0, gt_idx) / cfg.scale)

    # Masked cls loss: segments whose ray already hits the inner shell at the end of the segment are
    # classified as "hit" by the renderer (it ORs the inner-shell hit into the network output),
    # so the network does not need to learn them. Leave them out of the BCE.
    with torch.no_grad():
        seg_len = (exit_points - enter_points).norm(dim=1)
        inner_hit_mask, inner_hit_t, *_ = inner_mesh.ray_tracer.trace(
            enter_points + directions * eps, directions, allow_negative=True)
        inner_hit = inner_hit_mask & ((inner_hit_t - seg_len).abs() < 2e-3 * cfg.scale + 1e-3)
        keep = (~inner_hit).float()
    bce = F.binary_cross_entropy_with_logits(pred_logit, gt_mask.float(), reduction="none")

    gn = gt_normals.index_select(0, gt_idx)

    # per-segment cls weights (renormalized to mean 1 over the kept set, detached)
    cls_w = keep.clone()
    inc_max = getattr(cfg.train, "incidence_weight_max", 0.0)
    if inc_max > 0:
        # up-weight grazing hits: w = clamp(1 / (cos + 0.05), max), cos = -d . n_gt
        with torch.no_grad():
            inc_cos = (-(gn * directions.index_select(0, gt_idx)).sum(1)).clamp_min(0)
            inc_w = (1.0 / (inc_cos + 0.05)).clamp_max(inc_max)
            seg_w = torch.ones_like(keep)
            seg_w[gt_idx] = inc_w
            cls_w = cls_w * seg_w
    focal_gamma = getattr(cfg.train, "focal_gamma", 0.0)
    if focal_gamma > 0:
        with torch.no_grad():
            p_true = torch.sigmoid(pred_logit)
            p_true = torch.where(gt_mask, p_true, 1 - p_true)
            cls_w = cls_w * (1 - p_true).pow(focal_gamma)
    n_keep = keep.sum().clamp_min(1.0)
    cls_w = cls_w / (cls_w.sum() / n_keep).clamp_min(1e-12)
    cls_loss = (bce * cls_w).sum() / n_keep

    pn = pred_normals.index_select(0, gt_idx)
    if getattr(cfg.train, "normal_residual", False):
        # baseline = outer shell normal at segment entry; network predicts the adjustment
        bn = outer_normals.index_select(0, gt_idx)
        pn = bn + pn
        if log_baseline:
            with torch.no_grad():
                print(f"[normal_residual] baseline-only: l1={F.l1_loss(bn, gn).item():.4f} "
                      f"cos={(1 - F.cosine_similarity(bn, gn, dim=1)).mean().item():.4f}", flush=True)
    if getattr(cfg.train, "normal_loss_type", "l1") == "cosine":
        # renderer normalizes the predicted normal, so only direction matters
        normal_err = 1 - F.cosine_similarity(pn, gn, dim=1, eps=1e-6)
    else:
        normal_err = (pn - gn).abs().mean(1)
    if inc_max > 0:
        normal_loss = (normal_err * (inc_w / inc_w.mean())).mean()
    else:
        normal_loss = normal_err.mean()

    w = cfg.train.loss_weights
    # the factor 3 on cls is historical; benchmark configs were tuned with it
    loss = cls_loss * w["cls_loss"] * 3 + normal_loss * w["normal_loss"] + distance_loss * w["distance_loss"] + color_loss * w["color_loss"]
    return loss, normal_loss, distance_loss, cls_loss, color_loss


def train_model(cfg, fine_mesh, outer_mesh, inner_mesh, model, averaged_model, optimizer, scheduler, swa_scheduler, writer, model_config, run_name, step):
    prefetcher = BatchPrefetcher(cfg, fine_mesh, outer_mesh, inner_mesh)
    next_batch = prefetcher.prepare() if step < cfg.train.epochs else None
    pending_log = None  # previous epoch's tensorboard scalars, written one epoch late
    pinned_vals = [torch.empty(5, dtype=torch.float32, pin_memory=True) for _ in range(2)]

    progress = tqdm(range(step, cfg.train.epochs))
    for epoch in progress:
        model.train()
        batch = prefetcher.hand_over(next_batch)
        loss, normal_loss, distance_loss, cls_loss, color_loss = compute_losses(
            cfg, model, inner_mesh, batch, log_baseline=(epoch == step))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if cfg.train.use_averaged_model in ['EMA', 'SWA']:
            averaged_model.update_parameters(model)
        else:
            averaged_model = model
        if cfg.train.use_averaged_model == 'SWA':
            swa_scheduler.step()
        scheduler.step()

        # everything above is queued asynchronously; build the next batch on the
        # side stream while the GPU executes this step
        next_batch = prefetcher.prepare() if epoch + 1 < cfg.train.epochs else None

        mse = 0
        psnr = 0
        accuracy = 0
        flip = 0
        info = ""
        if epoch % cfg.train.evaluation_interval == cfg.train.evaluation_interval - 1:
            save_checkpoint(cfg, model_config, model, averaged_model, optimizer, scheduler, run_name, epoch + 1)
            if cfg.train.evaluate:
                if cfg.visualization.use_neural_renderer:
                    psnr, flip = render_predictions_with_neural_renderer(cfg, run_name)
                    info += f"PSNR={psnr:.4f}, FLIP={flip:.4f}, "
                else:
                    mse, psnr, accuracy = eval_model(cfg, fine_mesh, outer_mesh, inner_mesh, averaged_model)
                    info += f"MSE={mse:.6f}, PSNR={psnr:.4f}, "
            info += f"dist={distance_loss.item():.4f}, entr={cls_loss.item():.4f}, norm={normal_loss.item():.4f}, clr={color_loss.item():.4f}, total={loss.item():.4f}"
        if info != "":
            progress.set_postfix_str(info)
        if cfg.train.tensorboard:
            # copy this epoch's losses to pinned host memory without waiting, and
            # write the previous epoch's scalars (whose copy has finished by now)
            host_vals = pinned_vals[epoch % 2]
            host_vals.copy_(torch.stack([loss, normal_loss, distance_loss, cls_loss, color_loss]).detach(), non_blocking=True)
            event = torch.cuda.Event()
            event.record()
            if pending_log is not None:
                write_losses_to_tensorboard(writer, pending_log, cfg.train.evaluation_interval)
            pending_log = (epoch, event, host_vals, scheduler.get_last_lr()[0], mse, psnr, accuracy, flip,
                           epoch * cfg.train.sample_size)

    if pending_log is not None:
        write_losses_to_tensorboard(writer, pending_log, cfg.train.evaluation_interval)

    return model


def main(cfg):
    if cfg.train.model_start_checkpoint:
        checkpoint = torch.load(cfg.train.model_start_checkpoint, weights_only=False)
        model_config = checkpoint["model_config"]
    else:
        checkpoint = None
        model_config = cfg.model

    scale = cfg.scale
    fine_mesh = MeshWrapper.from_file(cfg.fine_mesh_path, n_max_samples=cfg.mesh_n_max_samples, scale=scale)
    outer_mesh = MeshWrapper.from_file(cfg.outer_mesh_path, n_max_samples=cfg.mesh_n_max_samples, scale=scale)
    inner_mesh = MeshWrapper.from_file(cfg.inner_mesh_path, n_max_samples=cfg.mesh_n_max_samples, scale=scale)

    os.makedirs(cfg.visualization.preview_path, exist_ok=True)
    os.makedirs(cfg.visualization.render_path, exist_ok=True)
    os.makedirs(cfg.train.checkpoints_path, exist_ok=True)

    save_mesh_previews({
        f"{cfg.visualization.preview_path}/{cfg.visualization.fine_mesh_preview_name}":  fine_mesh.mesh,
        f"{cfg.visualization.preview_path}/{cfg.visualization.outer_mesh_preview_name}": outer_mesh.mesh,
        f"{cfg.visualization.preview_path}/{cfg.visualization.inner_mesh_preview_name}": inner_mesh.mesh,
    }, cfg.visualization.image_size)

    model = RayModel(model_config, outer_mesh.mesh).to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.learning_rate)
    lr_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=cfg.train.learning_rate_scheduler_total_iters,
        eta_min=cfg.train.learning_rate * cfg.train.learning_rate_scheduler_min,
    )

    swa_scheduler = None
    if cfg.train.use_averaged_model == 'EMA':
        averaged_model = AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(cfg.train.ema_decay))
    elif cfg.train.use_averaged_model == 'SWA':
        averaged_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=cfg.train.swa_learing_rate)
    else:
        averaged_model = None

    step = 0
    if checkpoint:
        print("Loading model state...")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["learning_rate_scheduler"])
        step = checkpoint["step"]

        if cfg.train.use_averaged_model in ['EMA', 'SWA']:
            averaged_model.load_state_dict(checkpoint["averaged_model"])
    
    writer = None
    run_name = f"run_{int(time.time())}"
    if cfg.train.run_name:
        run_name = cfg.train.run_name
    if cfg.train.tensorboard:
        writer = SummaryWriter(f"{cfg.train.tensorboard_path}/{run_name}")

    model = train_model(
        cfg, fine_mesh, outer_mesh, inner_mesh,
        model, averaged_model, optimizer, 
        lr_scheduler, swa_scheduler, 
        writer, model_config, run_name, step
    )

    save_checkpoint(cfg, model_config, model, averaged_model, optimizer, lr_scheduler, run_name, cfg.train.epochs)
    

if __name__ == "__main__":
    import config as cfg_module
    cfg = cfg_module.cfg

    """
    apply override
    expect json of format

    {
        "scale": 100,
        "fine_mesh_path": "models/foo.fbx",
        "model": {
            "network_config": {"n_neurons": 64}
        },
        "train": {
            "run_name": "my_run",
            "checkpoints_path": "checkpoints"
        }
    }
    """

    if len(sys.argv) > 1:
        print(f"Applying overrides from {sys.argv[1]}...")
        with open(sys.argv[1]) as f:
            overrides = json.load(f)
        def apply_overrides(ns, d):
            for key, val in d.items():
                if isinstance(val, dict) and hasattr(ns, key) and not isinstance(getattr(ns, key), dict):
                    apply_overrides(getattr(ns, key), val)
                else:
                    setattr(ns, key, val)
        apply_overrides(cfg, overrides)

    print("Configuration:")
    print(cfg)

    main(cfg)
