import os
from typing import Optional, Tuple, List, Union, Callable

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import trange
import pytorch_lightning as pl
import yaml
from data.nerf_data import NeRFDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from einops import rearrange, repeat


class PositionalEncoder(nn.Module):
    r"""
    Sine-cosine positional encoder for input points.
    """

    def __init__(self, d_input: int, n_freqs: int, log_space: bool = False):
        super().__init__()
        self.d_input = d_input
        self.n_freqs = n_freqs
        self.log_space = log_space
        self.d_output = d_input * (1 + 2 * self.n_freqs)
        self.embed_fns = [lambda x: x]

        # Define frequencies in either linear or log scale
        if self.log_space:
            freq_bands = 2.0 ** torch.linspace(0.0, self.n_freqs - 1, self.n_freqs)
        else:
            freq_bands = torch.linspace(
                2.0**0.0, 2.0 ** (self.n_freqs - 1), self.n_freqs
            )

        # Alternate sin and cos
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            self.embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))

    def forward(self, x) -> torch.Tensor:
        r"""
        Apply positional encoding to input.
        """
        return torch.concat([fn(x) for fn in self.embed_fns], dim=-1)


class NeRFModule(nn.Module):
    def __init__(self, d_input, n_layers, d_filter, skip, d_viewdirs):
        r"""
        nerf network.
        d_input: encoded points dim
        n_layers: how many mlp will be used
        d_filter: can be understanded as hidden_dims
        skip: layers will be merged
        d_viewdirs: encoded viewdirs
        """
        super().__init__()
        self.d_input = d_input
        self.skip = skip
        self.act = nn.functional.relu
        self.d_viewdirs = d_viewdirs

        # Create model layers
        self.layers = nn.ModuleList(
            [nn.Linear(self.d_input, d_filter)]
            + [
                (
                    nn.Linear(d_filter + self.d_input, d_filter)
                    if i in skip
                    else nn.Linear(d_filter, d_filter)
                )
                for i in range(n_layers - 1)
            ]
        )

        # Bottleneck layers
        if self.d_viewdirs is not None:
            # If using viewdirs, split alpha and RGB
            self.alpha_out = nn.Linear(d_filter, 1)
            self.rgb_filters = nn.Linear(d_filter, d_filter)
            self.branch = nn.Linear(d_filter + self.d_viewdirs, d_filter // 2)
            self.output = nn.Linear(d_filter // 2, 3)
        else:
            # If no viewdirs, use simpler output
            self.output = nn.Linear(d_filter, 4)

    def forward(self, x, viewdirs):
        r"""
        x: (chunksize, d_input)
        viewdirs: (chunksize, d_viewdirs)

        return:
            (chunksize, 4)
        """
        # Cannot use viewdirs if instantiated with d_viewdirs = None
        if self.d_viewdirs is None and viewdirs is not None:
            raise ValueError("Cannot input x_direction if d_viewdirs was not given.")

        # Apply forward pass up to bottleneck
        x_input = x
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x))
            if i in self.skip:
                x = torch.cat([x, x_input], dim=-1)

        # Apply bottleneck
        if self.d_viewdirs is not None:
            # Split alpha from network output
            alpha = self.alpha_out(x)

            # Pass through bottleneck to get RGB
            x = self.rgb_filters(x)
            x = torch.concat([x, viewdirs], dim=-1)
            x = self.act(self.branch(x))
            x = self.output(x)

            # Concatenate alphas to output
            x = torch.concat([x, alpha], dim=-1)
        else:
            # Simple output
            x = self.output(x)
        return x


class NeRF(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.use_viewdirs = config["encoder"]["use_viewdirs"]
        self.use_fine_model = config["model"]["use_fine_model"]
        # Create encoders for points and view directions
        self.encoder = PositionalEncoder(
            config["encoder"]["d_input"],
            config["encoder"]["n_freqs"],
            config["encoder"]["log_space"],
        )
        self.viewdirs_encoder = (
            PositionalEncoder(
                config["encoder"]["d_input"], config["encoder"]["n_freqs_views"]
            )
            if self.use_viewdirs
            else None
        )
        self.d_viewdirs = self.viewdirs_encoder.d_output if self.use_viewdirs else None
        # create model
        self.coarse_model = NeRFModule(
            self.encoder.d_output,
            config["model"]["n_layers"],
            config["model"]["d_filter"],
            config["model"]["skip"],
            self.d_viewdirs,
        )
        self.fine_model = (
            NeRFModule(
                self.encoder.d_output,
                config["model"]["n_layers_fine"],
                config["model"]["d_filter_fine"],
                config["model"]["skip"],
                self.d_viewdirs,
            )
            if self.use_fine_model
            else None
        )

        self.near = config["model"]["near"]
        self.far = config["model"]["far"]

    def sample_stratified(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: float,
        far: float,
        n_samples: int,
        perturb: Optional[bool] = True,
        inverse_depth: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Sample along ray from regularly-spaced bins.

        rays_o: (all_chunksize, 3)
        rays_d: (all_chunksize, 3)

        return:
            pts: (all_chunksize, n_samples, 3)
            z_vals: (all_chunksize, n_samples)
        """

        # Grab samples for space integration along ray
        t_vals = torch.linspace(0.0, 1.0, n_samples, device=rays_o.device)
        if not inverse_depth:
            # Sample linearly between `near` and `far`
            z_vals = near * (1.0 - t_vals) + far * (t_vals)
        else:
            # Sample linearly in inverse depth (disparity)
            z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * (t_vals))

        # Draw uniform samples from bins along ray
        # if preturb is True, so add some noise into samples
        if perturb:
            mids = 0.5 * (z_vals[1:] + z_vals[:-1])
            upper = torch.concat([mids, z_vals[-1:]], dim=-1)
            lower = torch.concat([z_vals[:1], mids], dim=-1)
            t_rand = torch.rand([n_samples], device=z_vals.device)  # noise
            z_vals = lower + (upper - lower) * t_rand  # add noise
        # trick: expand z_vals like rays_o, so the samples can be mapped to specific ray
        z_vals = z_vals.expand(list(rays_o.shape[:-1]) + [n_samples])

        # Apply scale from `rays_d` and offset from `rays_o` to samples
        # pts: (width, height, n_samples, 3)
        # rays_o (chunksize, 3) rays_d (chunksize, 3) z_vals (chunksize, n_samples)
        # print(f"sample_stratified->{rays_o.shape}, {rays_d.shape}, {z_vals.shape}")
        r_o = repeat(rays_o, "c d -> c 1 d")
        r_d = repeat(rays_d, "c d -> c 1 d")
        z_v = repeat(z_vals, "c n -> c n 1")
        pts = r_o + r_d + z_v
        # print(f"sample_stratified->{pts.shape}")
        return pts, z_vals

    def cumprod_exclusive(self, tensor: torch.Tensor) -> torch.Tensor:
        r"""
        (Courtesy of https://github.com/krrish94/nerf-pytorch)

        Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

        Args:
        tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
            is to be computed.
        Returns:
        cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
            tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
        """

        # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
        cumprod = torch.cumprod(tensor, -1)
        # "Roll" the elements along dimension 'dim' by 1 element.
        cumprod = torch.roll(cumprod, 1, -1)
        # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
        cumprod[..., 0] = 1.0
        return cumprod

    def raw2outputs(
        self,
        raw: torch.Tensor,
        z_vals: torch.Tensor,
        rays_d: torch.Tensor,
        raw_noise_std: float = 0.0,
        white_bkgd: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Volume Rendering main process.
        Convert the raw NeRF output into RGB and other maps.
        """

        # Difference between consecutive elements of `z_vals`. [n_rays, n_samples]
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1)

        # Multiply each distance by the norm of its corresponding direction ray
        # to convert to real world distance (accounts for non-unit directions).
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

        # Add noise to model's predictions for density. Can be used to
        # regularize network during training (prevents floater artifacts).
        noise = 0.0
        if raw_noise_std > 0.0:
            noise = torch.randn(raw[..., 3].shape) * raw_noise_std

        # Predict density of each sample along each ray. Higher values imply
        # higher likelihood of being absorbed at this point. [n_rays, n_samples]
        alpha = 1.0 - torch.exp(-nn.functional.relu(raw[..., 3] + noise) * dists)

        # Compute weight for RGB of each sample along each ray. [n_rays, n_samples]
        # The higher the alpha, the lower subsequent weights are driven.
        weights = alpha * self.cumprod_exclusive(1.0 - alpha + 1e-10)

        # Compute weighted RGB map.
        rgb = torch.sigmoid(raw[..., :3])  # [n_rays, n_samples, 3]
        rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # [n_rays, 3]

        # Estimated depth map is predicted distance.
        depth_map = torch.sum(weights * z_vals, dim=-1)

        # Disparity map is inverse depth.
        disp_map = 1.0 / torch.max(
            1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1)
        )

        # Sum of weights along each ray. In [0, 1] up to numerical error.
        acc_map = torch.sum(weights, dim=-1)

        # To composite onto a white background, use the accumulated alpha map.
        if white_bkgd:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])

        return rgb_map, depth_map, acc_map, weights

    def sample_pdf(
        self,
        bins: torch.Tensor,
        weights: torch.Tensor,
        n_samples: int,
        perturb: bool = False,
    ) -> torch.Tensor:
        r"""
        Apply inverse transform sampling to a weighted set of points.
        """

        # Normalize weights to get PDF.
        pdf = (weights + 1e-5) / torch.sum(
            weights + 1e-5, -1, keepdims=True
        )  # [n_rays, weights.shape[-1]]

        # Convert PDF to CDF.
        cdf = torch.cumsum(pdf, dim=-1)  # [n_rays, weights.shape[-1]]
        cdf = torch.concat(
            [torch.zeros_like(cdf[..., :1]), cdf], dim=-1
        )  # [n_rays, weights.shape[-1] + 1]

        # Take sample positions to grab from CDF. Linear when perturb == 0.
        if not perturb:
            u = torch.linspace(0.0, 1.0, n_samples, device=cdf.device)
            u = u.expand(list(cdf.shape[:-1]) + [n_samples])  # [n_rays, n_samples]
        else:
            u = torch.rand(
                list(cdf.shape[:-1]) + [n_samples], device=cdf.device
            )  # [n_rays, n_samples]

        # Find indices along CDF where values in u would be placed.
        u = u.contiguous()  # Returns contiguous tensor with same values.
        inds = torch.searchsorted(cdf, u, right=True)  # [n_rays, n_samples]

        # Clamp indices that are out of bounds.
        below = torch.clamp(inds - 1, min=0)
        above = torch.clamp(inds, max=cdf.shape[-1] - 1)
        inds_g = torch.stack([below, above], dim=-1)  # [n_rays, n_samples, 2]

        # Sample from cdf and the corresponding bin centers.
        matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
        cdf_g = torch.gather(
            cdf.unsqueeze(-2).expand(matched_shape), dim=-1, index=inds_g
        )
        bins_g = torch.gather(
            bins.unsqueeze(-2).expand(matched_shape), dim=-1, index=inds_g
        )

        # Convert samples to ray length.
        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
        t = (u - cdf_g[..., 0]) / denom
        samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

        return samples  # [n_rays, n_samples]

    def sample_hierarchical(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        z_vals: torch.Tensor,
        weights: torch.Tensor,
        n_samples: int,
        perturb: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Apply hierarchical sampling to the rays.
        """

        # Draw samples from PDF using z_vals as bins and weights as probabilities.
        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        new_z_samples = self.sample_pdf(
            z_vals_mid, weights[..., 1:-1], n_samples, perturb=perturb
        )
        new_z_samples = new_z_samples.detach()

        # Resample points from ray based on PDF.
        z_vals_combined, _ = torch.sort(
            torch.cat([z_vals, new_z_samples], dim=-1), dim=-1
        )
        pts = (
            rays_o[..., None, :] + rays_d[..., None, :] * z_vals_combined[..., :, None]
        )  # [N_rays, N_samples + n_samples, 3]
        return pts, z_vals_combined, new_z_samples

    def get_chunks(
        self, inputs: torch.Tensor, chunksize: int = 2**15
    ) -> List[torch.Tensor]:
        r"""
        Divide an input into chunks.
        """
        return [inputs[i : i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

    def prepare_chunks(
        self, points: torch.Tensor, chunksize: int = 2**15
    ) -> List[torch.Tensor]:
        r"""
        Encode and chunkify points to prepare for NeRF model.
        """
        points = points.reshape((-1, 3))
        points = self.encoder(points)  # etc: (pts num, 63)
        points = self.get_chunks(points, chunksize=chunksize)
        return points

    def prepare_viewdirs_chunks(
        self, points: torch.Tensor, rays_d: torch.Tensor, chunksize: int = 2**15
    ) -> List[torch.Tensor]:
        r"""
        Encode and chunkify viewdirs to prepare for NeRF model.
        """
        # print(f"prepare_viewdirs_chunks-> {rays_d.shape}")
        # Prepare the viewdirs
        viewdirs = rays_d / torch.norm(
            rays_d, dim=-1, keepdim=True
        )  # normalize ray direction can get viewdirs
        # print(f"prepare_viewdirs_chunks-> {viewdirs.shape}")
        viewdirs = viewdirs[:, None, ...].expand(points.shape).reshape((-1, 3))
        # print(f"prepare_viewdirs_chunks-> {viewdirs.shape}")
        viewdirs = self.viewdirs_encoder(viewdirs)
        viewdirs = self.get_chunks(viewdirs, chunksize=chunksize)
        return viewdirs

    def forward(self, rays_o, rays_d):
        r"""
        Forward pass with optional view direction.
        """
        # 1. sample query points along each ray
        query_points, z_vals = self.sample_stratified(
            rays_o=rays_o,
            rays_d=rays_d,
            near=self.near,
            far=self.far,
            n_samples=self.config["stratified_sampling"]["n_samples"],
            perturb=self.config["stratified_sampling"]["perturb"],
            inverse_depth=self.config["stratified_sampling"]["inverse_depth"],
        )

        # 2. prepare batches
        batches = self.prepare_chunks(
            query_points, chunksize=self.config["train"]["chunksize"]
        )
        if self.viewdirs_encoder is not None:
            batches_viewdirs = self.prepare_viewdirs_chunks(
                query_points, rays_d, chunksize=self.config["train"]["chunksize"]
            )
        else:
            batches_viewdirs = [None] * len(batches)
        # check batch: batches->torch.Size([16384, 63]) batches_viewdirs->torch.Size([16384, 27])
        # 3. coarse model forward
        predictions = []
        for batch, batch_viewdirs in zip(batches, batches_viewdirs):
            predictions.append(self.coarse_model(batch, viewdirs=batch_viewdirs))
        raw = torch.cat(predictions, dim=0)
        # raw (all_chunksize, 4) query_points (all_chunksize, n_samples, 3)
        raw = rearrange(
            raw,
            "(c n) rgbd -> c n rgbd",
            c=query_points.shape[0],
            n=query_points.shape[1],
        )
        # raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])

        # Perform differentiable volume rendering to re-synthesize the RGB image.
        rgb_map, depth_map, acc_map, weights = self.raw2outputs(raw, z_vals, rays_d)
        # rgb_map, depth_map, acc_map, weights = render_volume_density(raw, rays_o, z_vals)
        outputs = {"z_vals_stratified": z_vals}

        # 4. fine model forward
        # Fine model pass.
        if self.config["hierarchical_sampling"]["n_samples_hierarchical"] > 0:
            # Save previous outputs to return.
            rgb_map_0, depth_map_0, acc_map_0 = rgb_map, depth_map, acc_map

            # Apply hierarchical sampling for fine query points.
            query_points, z_vals_combined, z_hierarch = self.sample_hierarchical(
                rays_o,
                rays_d,
                z_vals,
                weights,
                n_samples=self.config["hierarchical_sampling"][
                    "n_samples_hierarchical"
                ],
                perturb=self.config["hierarchical_sampling"]["perturb_hierarchical"],
            )

            # Prepare inputs as before.
            batches = self.prepare_chunks(
                query_points, chunksize=self.config["train"]["chunksize"]
            )
            if self.viewdirs_encoder is not None:
                batches_viewdirs = self.prepare_viewdirs_chunks(
                    query_points, rays_d, chunksize=self.config["train"]["chunksize"]
                )
            else:
                batches_viewdirs = [None] * len(batches)

            # Forward pass new samples through fine model.
            fine_model = (
                self.fine_model if self.fine_model is not None else self.coarse_model
            )
            predictions = []
            for batch, batch_viewdirs in zip(batches, batches_viewdirs):
                predictions.append(fine_model(batch, viewdirs=batch_viewdirs))
            raw = torch.cat(predictions, dim=0)
            # raw = raw.reshape(list(query_points.shape[:2]) + [raw.shape[-1]])
            raw = rearrange(
                raw,
                "(c n) rgbd -> c n rgbd",
                c=query_points.shape[0],
                n=query_points.shape[1],
            )

            # Perform differentiable volume rendering to re-synthesize the RGB image.
            rgb_map, depth_map, acc_map, weights = self.raw2outputs(
                raw, z_vals_combined, rays_d
            )

            # Store outputs.
            outputs["z_vals_hierarchical"] = z_hierarch
            outputs["rgb_map_0"] = rgb_map_0
            outputs["depth_map_0"] = depth_map_0
            outputs["acc_map_0"] = acc_map_0
        # Store outputs.
        outputs["rgb_map"] = rgb_map
        outputs["depth_map"] = depth_map
        outputs["acc_map"] = acc_map
        outputs["weights"] = weights
        return outputs

    def training_step(self, batch, batch_idx):
        rays_o, rays_d, target_img = batch
        # remove batchsize because batchsize is constantly 1
        rays_o = rays_o[0]
        rays_d = rays_d[0]
        target_img = target_img[0]
        outputs = self.forward(rays_o, rays_d)
        rgb_predicted = outputs["rgb_map"]
        loss = torch.nn.functional.mse_loss(rgb_predicted, target_img.reshape(-1, 3))
        self.log(f"train_loss", loss, prog_bar=True)
        if batch_idx % 10 == 0:
            self.logger.experiment.add_image(
                "train/target_img",
                target_img.reshape(100, 100, 3).permute(2, 0, 1),
                self.global_step,
            )
            self.logger.experiment.add_image(
                "train/predict_img",
                rgb_predicted.reshape(100, 100, 3).permute(2, 0, 1),
                self.global_step,
            )

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(), lr=self.config["optimizer"]["lr"]
        )
        return optimizer


if __name__ == "__main__":
    pl.seed_everything(7)

    # read yaml config file
    with open("configs/nerf.yaml", "r") as file:
        cfg = yaml.safe_load(file)

    # dataloader = get_mnist_train_loader(cfg["train"]["batch_size"])
    # dm = Cifar10DataModule(batch_size=cfg["train"]["batch_size"])
    dm = NeRFDataModule()

    model = NeRF(cfg)

    trainer = pl.Trainer(
        max_epochs=10000,
        accelerator="gpu",
        devices=[0],
        logger=TensorBoardLogger("logs/", name="nerf"),
        callbacks=[LearningRateMonitor(logging_interval="step")],
    )
    trainer.fit(model, datamodule=dm)
    # trainer.test(model, datamodule=dm)

    # save last ckpt
    # output_dir = cfg["train"]["output_dir"]
    # model_name = cfg["model"]["name"]
    # final_dir_path = f"{output_dir}/{model_name}"
    # if not os.path.exists(final_dir_path):
    #     os.makedirs(final_dir_path)
    # torch.save(model.state_dict(), f"{final_dir_path}/last_checkpoint.pth")
    # pprint(
    #     f"train process complete. save checkpoint at {final_dir_path}/last_checkpoint.pth"
    # )
