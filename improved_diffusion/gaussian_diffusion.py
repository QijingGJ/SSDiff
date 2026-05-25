"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from kornia.filters import sobel
import lpips
from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood
from .dpm_solver_sampler import NoiseScheduleVP, model_wrapper, DPM_Solver


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    # 余弦调度 s = 0.008
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    PREVIOUS_X = enum.auto()  # the model predicts x_{t-1}
    START_X = enum.auto()  # the model predicts x_0
    EPSILON = enum.auto()  # the model predicts epsilon


class ModelVarType(enum.Enum):
    """
    What is used as the model's output variance.

    The LEARNED_RANGE option has been added to allow the model to predict
    values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    """

    LEARNED = enum.auto()
    FIXED_SMALL = enum.auto()
    FIXED_LARGE = enum.auto()
    LEARNED_RANGE = enum.auto()


class LossType(enum.Enum):
    MSE = enum.auto()  # use raw MSE loss (and KL when learning variances)
    RESCALED_MSE = (
        enum.auto()
    )  # use raw MSE loss (with RESCALED_KL when learning variances)
    KL = enum.auto()  # use the variational lower-bound
    RESCALED_KL = enum.auto()  # like KL, but rescale to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    :param model_mean_type: a ModelMeanType determining what the model outputs.
    :param model_var_type: a ModelVarType determining how variance is output.
    :param loss_type: a LossType determining the loss function to use.
    :param rescale_timesteps: if True, pass floating point timesteps into the
                              model so that they are always scaled like in the
                              original paper (0 to 1000).
    """

    def __init__(
            self,
            *,
            betas,
            model_mean_type,
            model_var_type,
            loss_type,
            rescale_timesteps=False,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        self.rescale_timesteps = rescale_timesteps

        # Use float64 for accuracy.
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)  # 1/根号下α^
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)  # 1/根号下α^ - 1 = (1-根号下α^)/根号下α^

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
                betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
                betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
                (1.0 - self.alphas_cumprod_prev)
                * np.sqrt(alphas)
                / (1.0 - self.alphas_cumprod)
        )

        # self.ssim = pytorch_ssim.SSIM(window_size=11)
        # self.ssim_loss = SSIM()
        # self.loss_dice = DiceLoss(n_classes=4)
        self.lpips_model = lpips.LPIPS(net='alex', verbose=False).to('cuda:0')

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
                _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
                + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
                * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
                posterior_mean.shape[0]
                == posterior_variance.shape[0]
                == posterior_log_variance_clipped.shape[0]
                == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
            self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, model_output=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        # model: improved_diffusion.respace._WrappedModel
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)

        # -------------------------
        if C != 1:
            C = 1
        if x.shape[1] != 1:
            x = x[:, -1:, ...]  # 前向分布只在最后一个通道noise进行
        # -------------------------

        # ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE 预测方差
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            # 预测可学习方差
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            # 预测可学习方差范围
            else:
                # log βt^
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                # log βt
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                # exp(v * log βt + (1 - v) * log βt^)
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        # ModelMeanType.EPSILON  预测噪声
        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output
        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            # 直接预测x0
            if self.model_mean_type == ModelMeanType.START_X:
                pred_xstart = process_xstart(model_output)
            # 预测的是噪声
            else:
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape  # [1,1,128,128]
        )
        return {
            "mean": model_mean,  # 分布 q(xt-1) 的均值
            "variance": model_variance,  # 分布 q(xt-1) 的方差
            "log_variance": model_log_variance,  # 分布 q(xt-1) 的对数方差
            "pred_xstart": pred_xstart,
            "eps": model_output,  # 预测的噪声
        }

    def p_mean_variance_multi(
            self, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, model_output=None, pred_type=None):

        # x为xt
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]  # b,1
        assert t.shape == (B,)

        # ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE 预测方差
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)

            # 预测可学习方差
            if self.model_var_type == ModelVarType.LEARNED:
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            # 预测可学习方差范围
            else:
                # log βt^
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                # log βt
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # The model_var_values is [-1, 1] for [min_var, max_var].
                frac = (model_var_values + 1) / 2
                # exp(v * log βt + (1 - v) * log βt^)
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if pred_type == 'START_X':
            pred_xstart = process_xstart(model_output)
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        elif pred_type == 'EPSILON':
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output

        assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape  # [1,1,128,128]
        )
        return {
            "mean": model_mean,  # 分布 q(xt-1) 的均值
            "variance": model_variance,  # 分布 q(xt-1) 的方差
            "log_variance": model_log_variance,  # 分布 q(xt-1) 的对数方差
            "pred_xstart": pred_xstart,
            "eps": model_output,  # 预测的噪声
        }

    def p_mean_variance_multi_x0(
            self, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, model_output=None, pred_type=None):

        # x为xt
        if model_kwargs is None:
            model_kwargs = {}

        model_variance, model_log_variance = self.posterior_variance, self.posterior_log_variance_clipped,
        model_variance = _extract_into_tensor(model_variance, t, x.shape)
        model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        # model_output = th.softmax(model_output, dim=1)
        # model_output = th.argmax(model_output, dim=1).unsqueeze(1)

        if pred_type == 'START_X':
            pred_xstart = process_xstart(model_output)
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        elif pred_type == 'EPSILON':
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            )
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output

        assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape  # [1,1,128,128]
        )
        return {
            "mean": model_mean,  # 分布 q(xt-1) 的均值
            "variance": model_variance,  # 分布 q(xt-1) 的方差
            "log_variance": model_log_variance,  # 分布 q(xt-1) 的对数方差
            "pred_xstart": pred_xstart,
            "eps": model_output,  # 预测的噪声
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
                _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
                - _extract_into_tensor(
            self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
        ) * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
                _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def _predict_xstart_from_eps_plus(self, x_t, t, eps):
        """保持梯度流的x0预测"""
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x_t.shape)
        x0 = (x_t - eps * th.sqrt(1 - alpha_bar)) / th.sqrt(alpha_bar)
        return x0  # 保持自动微分

    def p_sample(
            self, model, x, t, pack, clip_denoised=True, denoised_fn=None, model_kwargs=None, index=None
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)
        slice_ID = pack['npz_name'][0].split('.')[0]

        out_1 = self.p_mean_variance_multi(
            x[:, -2:-1, ...],  # (1, 1, 128, 128)
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            model_output=model_output['pred_eps'],
            pred_type='EPSILON',
        )
        # 模型预测的是x0 不是xt-1, 故前一步的输出不能直接作为下一步的输入
        out_2 = self.p_mean_variance_multi(
            x[:, -1:, ...],  # (1, 1, 128, 128)
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            model_output=model_output['pred_mask'],
            pred_type='EPSILON',  # START_X
        )
        noise_lge = th.randn_like(x[:, -1:, ...])
        noise_msk = th.randn_like(x[:, -1:, ...])
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        sample_lge = out_1["mean"] + nonzero_mask * th.exp(0.5 * out_1["log_variance"]) * noise_lge  # 重参数化技巧
        sample_msk = out_2["mean"] + nonzero_mask * th.exp(0.5 * out_2["log_variance"]) * noise_msk

        return {"sample_lge": sample_lge,
                "sample_msk": sample_msk,
                "pred_xstart_lge": out_1["pred_xstart"],
                }

    def p_sample_cfg(
            self, model, x, t, pack, clip_denoised=True, denoised_fn=None, model_kwargs=None, index=None
    ):
        xc = x
        # xc[:, 3: 6, ...] = torch.zeros([1, 3, 128, 128])
        xc[:, 1: 3, ...] = torch.zeros([1, 2, 128, 128])

        with th.no_grad():
            # 无条件预测
            uncond_output = model(x, self._scale_timesteps(t), **model_kwargs)
            # 有条件预测
            cond_output = model(x, self._scale_timesteps(t), **model_kwargs)

        # 应用CFG公式合并预测结果
        model_output = {
            'pred_eps': uncond_output['pred_eps'] + 3 * (cond_output['pred_eps'] - uncond_output['pred_eps'])
        }
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            model_output=model_output['pred_eps']
        )

        noise = th.randn_like(x[:, -1:, ...])  # 噪声
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise  # 重参数化技巧

        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
            self,
            model,
            shape,
            img,
            pack,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress=True,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param clip_denoised: if True, clip x_start predictions to [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param device: if specified, the device to create the samples on.
                       If not specified, use a model parameter's device.
        :param progress: if True, show a tqdm progress bar.
        :return: a non-differentiable batch of samples.
        """
        noise = img  # cine+noise
        final = None
        for sample in self.p_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                pack=pack,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
        ):
            final = sample
        return torch.cat([final["sample_lge"], final["sample_msk"]], dim=1)
        # return final

    def p_sample_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            pack=None,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress=True,
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.

        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        if device is None:
            device = next(model.parameters()).device  # 普通模型
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        img = img.to(device)
        indices = list(range(self.num_timesteps))[::-1]  # 时间步取逆序

        org_c = img.size(1)  # cine的通道+1
        org_MRI = img[:, :-2, ...]  # cine图像

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        index = 999
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)  # 时间步 倒序时间步采样

            with th.no_grad():
                if img.size(1) != org_c:
                    img = th.cat((org_MRI, img), dim=1).float()

                out = self.p_sample(
                    model,
                    img,  # (1,8,128,128)
                    t,
                    pack,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    index=index,
                )  # out:[1,1,128,128]
                index = index - 1
                yield out  # must
                # img = out["sample_lge"]
                img = torch.cat([out['sample_lge'], out['sample_msk']], dim=1)  # out["sample"]  # must

    def dpm_solver_sampling(self, model,
                            shape,
                            img,
                            pack,
                            clip_denoised=True,
                            denoised_fn=None,
                            model_kwargs=None,
                            device=None,
                            classifier=None,
                            classifier_scale=0.0,
                            progress=True):

        def var_change(var_value, x_t, t):
            min_log = _extract_into_tensor(
                self.posterior_log_variance_clipped, t, x_t.shape
            )
            # log βt
            max_log = _extract_into_tensor(np.log(self.betas), t, x_t.shape)
            # The model_var_values is [-1, 1] for [min_var, max_var].
            frac = (var_value + 1) / 2
            # exp(v * log βt + (1 - v) * log βt^)
            model_log_variance = frac * max_log + (1 - frac) * min_log
            return model_log_variance

        def model_fn(x, t, **model_kwargs):
            out = model(x, t, **model_kwargs)
            out1 = torch.split(out['pred_eps'], 1, dim=1)[0]
            out2 = torch.split(out['pred_mask'], 1, dim=1)[0]
            return out1, out2

        def classifier_fn(x, t, y, **classifier_kwargs):
            logits = classifier(x, t)
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            return log_probs[range(len(logits)), y.view(-1)]

        # self.betas: [0.0001, 0.02]
        noise_schedule = NoiseScheduleVP(schedule='discrete', betas=torch.from_numpy(self.betas).float())
        model_fn_continuous = model_wrapper(
            model_fn,  # 函数, 输出预测 noise 均值 xs
            noise_schedule,
            model_type="noise",
            model_kwargs=model_kwargs,
            guidance_type="uncond" if classifier is None else "classifier",
            condition=model_kwargs["y"] if "y" in model_kwargs.keys() else None,
            guidance_scale=classifier_scale,
            classifier_fn=classifier_fn,
            classifier_kwargs={},
        )
        dpm_solver = DPM_Solver(
            model_fn_continuous,
            noise_schedule,
            algorithm_type='dpmsolver',  # dpmsolver++ // dpmsolver
        )

        x_sample = dpm_solver.sample(
            img,  # (1,8,128,128)
            steps=20,
            order=2,
            skip_type='logSNR',  # time_uniform//logSNR
            method='multistep',  # adaptive/multistep
            slice_ID=None,
        )
        return x_sample

    def p_mean_variance_simple(
            self, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, model_output=None, pred_type=None,
    ):
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)

        # -------------------------
        if C != 1:
            C = 1
        if x.shape[1] != 1:
            x = x[:, -1:, ...]  # 前向分布只在最后一个通道noise进行
        # -------------------------
        assert model_output.shape == (B, C * 2, *x.shape[2:])
        model_output, model_var_values = th.split(model_output, C, dim=1)
        # 预测可学习方差
        if self.model_var_type == ModelVarType.LEARNED:
            model_log_variance = model_var_values
            model_variance = th.exp(model_log_variance)
        # 预测可学习方差范围
        else:
            # log βt^
            min_log = _extract_into_tensor(
                self.posterior_log_variance_clipped, t, x.shape
            )
            # log βt
            max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
            # The model_var_values is [-1, 1] for [min_var, max_var].
            frac = (model_var_values + 1) / 2
            # exp(v * log βt + (1 - v) * log βt^)
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = th.exp(model_log_variance)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        # ModelMeanType.EPSILON  预测噪声
        if pred_type == 'x0':
            pred_xstart = process_xstart(model_output)
        # 预测的是噪声
        else:
            pred_xstart = process_xstart(
                self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
            )
        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_xstart, x_t=x, t=t
        )
        assert (
                model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape  # [1,1,128,128]
        )
        return {
            "mean": model_mean,  # 分布 q(xt-1) 的均值
            "variance": model_variance,  # 分布 q(xt-1) 的方差
            "log_variance": model_log_variance,  # 分布 q(xt-1) 的对数方差
            "pred_xstart": pred_xstart,
        }

    def p_combine_sample(
            self, model, x1, x2, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        out1 = self.p_mean_variance(
            model,
            x1,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        out2 = self.p_mean_variance(
            model,
            x2,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        noise = th.randn_like(x1[:, -1:, ...])  # 噪声
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x1.shape) - 1)))
        )

        model_mean = 0.3 * out1["mean"] + 0.7 * out2["mean"]
        log_var = 0.3 * out1["log_variance"] + 0.7 * out2["log_variance"]
        sample = model_mean + nonzero_mask * th.exp(0.5 * log_var) * noise
        return {"sample": sample, "pred_xstart_1": out1["pred_xstart"], }

    def p_combine_sample_progressive(self, model,
                                     shape,
                                     img_cine,
                                     img_t2,
                                     clip_denoised=True,
                                     denoised_fn=None,
                                     model_kwargs=None,
                                     device=None,
                                     progress=True, ):
        if device is None:
            device = next(model.parameters()).device

        img_cine = img_cine.to(device)
        img_t2 = img_t2.to(device)
        indices = list(range(self.num_timesteps))[::-1]

        org_c = img_cine.size(1)
        org_cine = img_cine[:, :-1, ...]
        org_t2 = img_t2[:, :-1, ...]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():

                if img_cine.size(1) != org_c:
                    img_cine = th.cat((org_cine, img_cine), dim=1)

                if img_t2.size(1) != org_c:
                    img_t2 = th.cat((org_t2, img_t2), dim=1)

                out = self.p_combine_sample(
                    model,
                    img_cine.float(),
                    img_t2.float(),
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )  # out:[1,1,128,128]
                yield out
                img_cine = out["sample"]
                img_t2 = out["sample"]

    def p_combine_sample_loop(self, model,
                              shape,
                              img_cine,
                              img_t2,
                              clip_denoised=True,
                              denoised_fn=None,
                              model_kwargs=None,
                              device=None,
                              progress=True, ):

        for sample in self.p_combine_sample_progressive(
                model,
                shape,
                img_cine,
                img_t2,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
        ):
            final = sample
        return final["sample"]

    def ddim_sample(
            self, model, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None, eta=0.0, t_next=None
    ):
        """
        Sample x_{t-1} from the model using DDIM.
        Same usage as p_sample().
        """
        # ---------------------------------------------------------------------------------------------
        # 参数计算
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, (1, 1, 128, 128))
        # alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, (1, 1, 128, 128))  # α^_(t-1)
        device = t.device
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod, th.tensor([t_next]).to(device), (1, 1, 128, 128))

        # eta=0 时 sigma=0
        sigma = (
                eta
                * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # ---------------------------------------------------------------------------------------------

        # 模型预测
        model_output = model(x, self._scale_timesteps(t), **model_kwargs)

        x1 = x[:, -2:-1, ...]
        x2 = x[:, -1:, ...]

        out_1 = self.p_mean_variance_multi(
            x1,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            model_output=model_output['pred_eps'],
            pred_type='EPSILON',
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.

        eps_1 = self._predict_eps_from_xstart(x1, t, out_1["pred_xstart"])  # 根据x0计算噪声

        out_2 = self.p_mean_variance_multi(
            x2,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            model_output=model_output['pred_mask'],
            pred_type='EPSILON',
        )  # (1, 1, 128, 128)
        eps_2 = self._predict_eps_from_xstart(x2, t, out_2["pred_xstart"])

        # Equation 12.
        noise_lge = th.randn_like(x1)  # ϵ
        noise_msk = th.randn_like(x2)

        mean_pred_lge = (
                out_1["pred_xstart"] * th.sqrt(alpha_bar_prev)
                + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps_1
        )

        mean_pred_msk = (
                out_2["pred_xstart"] * th.sqrt(alpha_bar_prev)
                + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps_2
        )

        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

        sample_lge = mean_pred_lge + nonzero_mask * sigma * noise_lge
        sample_msk = mean_pred_msk + nonzero_mask * sigma * noise_msk

        return {"sample_lge": sample_lge, "pred_xstart_lge": out_1["pred_xstart"],
                "sample_msk": sample_msk, "pred_xstart_msk": out_2["pred_xstart"]}

    def ddim_reverse_sample(
            self,
            model,
            x,
            t,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using DDIM reverse ODE.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = (_extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x - out["pred_xstart"]) / \
              _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)

        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Equation 12. reversed
        mean_pred = (
                out["pred_xstart"] * th.sqrt(alpha_bar_next)
                + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
            self,
            model,
            shape,
            img,
            pack,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress=True,
            eta=0.5,  # η
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        noise = img
        final = None
        for sample in self.ddim_sample_loop_progressive(
                model,
                shape,
                noise=noise,
                pack=pack,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
                device=device,
                progress=progress,
                eta=eta,
        ):
            final = sample
        return torch.cat([final["sample_lge"], final["sample_msk"]], dim=1)

    def ddim_sample_loop_progressive(
            self,
            model,
            shape,
            noise=None,
            pack=None,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None,
            device=None,
            progress=False,
            eta=0.0,  # η
            ddim_timesteps=20,  # 新增：自定义采样步数 S << T
            ddim_order=2  # 新增：子序列生成方式（1=线性，2=二次）
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            img = noise
        else:
            img = th.randn(*shape, device=device)

        img = img.to(device)
        indices = list(range(self.num_timesteps))[::-1]

        # -----------------------------------------------------------------------------------------------
        # ==== 关键修改1：生成DDIM子序列 ====
        if ddim_order == 1:
            # 线性子序列（论文默认）
            c = self.num_timesteps // ddim_timesteps
            subsequence = np.asarray(list(range(0, self.num_timesteps, c)))
        elif ddim_order == 2:
            # 二次序列（更早关注低频信息）
            subsequence = np.linspace(0, np.sqrt(self.num_timesteps * 0.8), ddim_timesteps) ** 2
            subsequence = np.unique(subsequence.astype(int))
        else:
            raise ValueError(f"Unsupported ddim_order: {ddim_order}")

        # 确保包含起点和终点
        subsequence = np.append(subsequence, self.num_timesteps - 1)
        subsequence = np.sort(subsequence)[::-1]  # 逆序：从大到小

        # ==== 关键修改2：替换完整序列为子序列 ====
        indices = subsequence.tolist()  # 示例：[999, 899, 799, ..., 99, 0]
        # -----------------------------------------------------------------------------------------------

        org_c = img.size(1)
        org_MRI = img[:, :-2, ...]

        tqdm_indices = indices
        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm
            tqdm_indices = tqdm(indices)

        idx = 0
        for i in tqdm_indices:

            t = th.tensor([i] * shape[0], device=device)
            # 计算下一步时间步（子序列中的下一个点）
            t_next = indices[idx + 1] if idx < len(indices) - 1 else 0
            idx = idx + 1

            with th.no_grad():
                # ------------------------------------------------
                if img.size(1) != org_c:
                    img = th.cat((org_MRI, img), dim=1)
                # ------------------------------------------------
                out = self.ddim_sample(
                    model,
                    img,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                    eta=eta,
                    t_next=t_next  # 必须添加到ddim_sample参数
                )
                yield out
                img = torch.cat([out['sample_lge'], out['sample_msk']], dim=1)

    def _vb_terms_bpd(
            self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None, model_output=None,
    ):
        """
        Get a term for the variational lower-bound.

        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.

        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs, model_output=model_output,
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    def training_losses(self, model, x_start, t, cur_step, label, seg_mask,
                        model_kwargs=None, noise=None, add_noise_lge=None, add_noise_msk=None):
        """
        Compute training losses for a single timestep.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            add_noise_lge = th.randn_like(x_start[:, -1:, ...])  # 加的噪声
            add_noise_lge_new = add_noise_lge + 0.1 * th.randn_like(add_noise_lge)
            add_noise_msk = th.randn_like(x_start[:, -1:, ...])

        def replace_with_noise(img_tensor, noise_prob=0.1):
            if torch.rand(1) < noise_prob:
                noise = torch.zeros_like(img_tensor)
                img_tensor = noise
            return img_tensor

        # ------------------------------------------------------------------------------
        lge = x_start[:, -1:, ...]
        cine = x_start[:, 0:1, ...]
        t2 = x_start[:, 1:2, ...]
        noise_lge = self.q_sample(lge, t, noise=add_noise_lge_new)  # 扰动图xt
        noise_msk = self.q_sample(seg_mask, t, noise=add_noise_msk)
        cine = replace_with_noise(cine.float())
        t2 = replace_with_noise(t2.float())
        model_x_t = th.cat([cine.float(), t2.float(), noise_lge.float(), noise_msk.float()], dim=1)
        model_kwargs = {'y': label}
        # ------------------------------------------------------------------------------

        def compute_loss(x_start, x_t, model_output, noise, prefix, pred_type, loss_type, model_var):
            terms = {}
            if loss_type == 'KL' or loss_type == 'RESCALED_KL':
                terms[f"{prefix}_loss"] = self._vb_terms_bpd(
                    model=model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                    model_output=model_output
                )["output"]
                if loss_type == 'RESCALED_KL':
                    terms[f"{prefix}_loss"] *= self.num_timesteps
            elif loss_type == 'MSE' or loss_type == 'RESCALED_MSE':
                if model_var:
                    B, C = model_x_t.shape[:2]
                    C = 1
                    assert model_output.shape == (B, C * 2, *model_x_t.shape[2:])
                    model_output, model_var_values = th.split(model_output, C, dim=1)
                    frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                    terms[f"{prefix}_vb"] = self._vb_terms_bpd(
                        model=lambda *args, r=frozen_out: r,
                        x_start=x_start,
                        x_t=x_t,
                        t=t,
                        clip_denoised=False,
                        model_output=frozen_out
                    )["output"]
                    if loss_type == 'RESCALED_MSE':
                        terms[f"{prefix}_vb"] *= self.num_timesteps / 1000.0

                if pred_type == 'START_X':
                    target = x_start
                elif pred_type == 'EPSILON':
                    target = noise
                else:
                    target = self.q_posterior_mean_variance(x_start=x_start, x_t=x_t, t=t)[0]

                assert model_output.shape == target.shape == x_start.shape
                terms[f"{prefix}_mse"] = mean_flat((target - model_output) ** 2)
                terms[f"{prefix}_mae"] = mean_flat(th.abs(target - model_output))
            else:
                raise NotImplementedError(self.loss_type)
            return terms

        # ------------------------------------------------------------------------------

        output = model(model_x_t, self._scale_timesteps(t), **model_kwargs)
        terms_lge = compute_loss(x_start=lge, x_t=noise_lge, model_output=output['pred_eps'],
                                 noise=add_noise_lge, prefix="lge", pred_type='EPSILON',
                                 loss_type='RESCALED_MSE', model_var=True)

        terms_msk = compute_loss(x_start=seg_mask, x_t=noise_msk, model_output=output['pred_mask'],
                                 noise=add_noise_msk, prefix="msk", pred_type='EPSILON',
                                 loss_type='RESCALED_MSE', model_var=True)

        terms = {}
        loss_lge = terms_lge["lge_mse"] + terms_lge.get("vb_lge", 0) + terms_lge["lge_mae"]
        loss_msk = terms_msk["msk_mse"] + terms_lge.get("vb_msk", 0) + terms_msk["msk_mae"]
        terms["loss"] = loss_lge + loss_msk
        terms.update(terms_lge)
        terms.update(terms_msk)
        return terms

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower-bound, measured in bits-per-dim,
        as well as other related quantities.

        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param clip_denoised: if True, clip denoised samples.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.

        :return: a dict containing the following keys:
                 - total_bpd: the total variational lower-bound, per batch element.
                 - prior_bpd: the prior term in the lower-bound.
                 - vb: an [N x T] tensor of terms in the lower-bound.
                 - xstart_mse: an [N x T] tensor of x_0 MSEs for each timestep.
                 - mse: an [N x T] tensor of epsilon MSEs for each timestep.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Calculate VLB term at the current timestep
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

