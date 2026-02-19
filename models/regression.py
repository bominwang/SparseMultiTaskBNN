import numpy as np
import torch
from dataclasses import dataclass
from .distributions import Normal, DenseNormal, LowRankNormal, DenseNormalPrec, get_parameterization
from collections.abc import Callable
import torch.nn as nn


# 计算高斯分布间的KL散度
def gaussian_kl(p, q_scale):
    feat_dim = p.mean.shape[-1]
    mse_term = (p.mean ** 2).sum(-1).sum(-1) / q_scale
    trace_term = (p.trace_covariance / q_scale).sum(-1)
    logdet_term = (feat_dim * np.log(q_scale) - p.logdet_covariance).sum(-1)

    return 0.5 * (mse_term + trace_term + logdet_term)  # 返回KL散度，忽略常数项


# 计算Gamma分布的KL散度
def gamma_kl(cov_dist, prior_dist):
    kl = torch.distributions.kl.kl_divergence(cov_dist, prior_dist)
    return (kl).sum(-1)


# 计算期望下的高斯KL散度
def expected_gaussian_kl(p, q_scale, cov_dist):
    cov_factor = cov_dist.concentration / cov_dist.rate

    feat_dim = p.mean.shape[-1]
    mse_term = (p.mean ** 2).sum(-1) / q_scale
    trace_term = (p.trace_covariance / q_scale).sum(-1)
    combined_mse_term = (cov_factor * mse_term).sum(-1)
    logdet_term = (feat_dim * np.log(q_scale) - p.logdet_covariance).sum(-1)

    return 0.5 * (combined_mse_term + trace_term + logdet_term)  # currently exclude constant


# 数据类用于保存返回值，包括预测分布、训练损失函数和验证损失函数
@dataclass
class VBLLReturn():
    predictive: Normal | DenseNormal | torch.distributions.studentT.StudentT
    train_loss_fn: Callable[[torch.Tensor], torch.Tensor]
    val_loss_fn: Callable[[torch.Tensor], torch.Tensor]
    ood_scores: None | Callable[[torch.Tensor], torch.Tensor] = None


class Regression(nn.Module):
    """
    Variational Bayesian Linear Regression

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    regularization_weight : float
        Weight on regularization term in ELBO
    parameterization : str
        Parameterization of covariance matrix. Currently supports {'dense', 'diagonal', 'lowrank', 'dense_precision'}
    prior_scale : float
        Scale of prior covariance matrix
    wishart_scale : float
        Scale of Wishart prior on noise covariance
    dof : float
        Degrees of freedom of Wishart prior on noise covariance


    变分贝叶斯线性回归
    ----------
    in_features : int
        输入特征数
    out_features : int
        输出特征数
    regularization_weight : float
        ELBO中的正则化项权重
    parameterization : str
        协方差矩阵的参数化方法，支持{'dense', 'diagonal', 'lowrank', 'dense_precision'}
    prior_scale : float
        先验协方差矩阵的尺度
    wishart_scale : float
        噪声协方差的Wishart先验尺度
    dof : float
        噪声协方差的Wishart先验自由度
    """

    def __init__(self,
                 in_features,
                 out_features,
                 regularization_weight,
                 parameterization='dense',
                 prior_scale=1.,
                 wishart_scale=1e-2,
                 cov_rank=None,
                 dof=1.):
        super(Regression, self).__init__()

        self.wishart_scale = wishart_scale
        self.dof = (dof + out_features + 1.) / 2.  # 计算自由度
        self.regularization_weight = regularization_weight

        # define prior, currently fixing zero mean and arbitrarily scaled cov
        # 定义先验，先验均值固定为0，协方差缩放
        self.prior_scale = prior_scale * (1. / in_features)

        # noise distribution # 噪声分布参数
        self.noise_mean = nn.Parameter(torch.zeros(out_features), requires_grad=False)
        self.noise_logdiag = nn.Parameter(torch.randn(out_features) * (np.log(wishart_scale)))

        # last layer distribution # 最后一层的分布参数
        self.W_dist = get_parameterization(parameterization)
        self.W_mean = nn.Parameter(torch.randn(out_features, in_features))

        # 根据协方差矩阵的参数化方式初始化参数
        if parameterization == 'diagonal':
            self.W_logdiag = nn.Parameter(torch.randn(out_features, in_features) - 0.5 * np.log(in_features))
        elif parameterization == 'dense':
            self.W_logdiag = nn.Parameter(torch.randn(out_features, in_features) - 0.5 * np.log(in_features))
            self.W_offdiag = nn.Parameter(torch.randn(out_features, in_features, in_features) / in_features)
        elif parameterization == 'dense_precision':
            self.W_logdiag = nn.Parameter(torch.randn(out_features, in_features) + 0.5 * np.log(in_features))
            self.W_offdiag = nn.Parameter(torch.randn(out_features, in_features, in_features) * 0.0)
        elif parameterization == 'lowrank':
            self.W_logdiag = nn.Parameter(torch.randn(out_features, in_features) - 0.5 * np.log(in_features))
            self.W_offdiag = nn.Parameter(torch.randn(out_features, in_features, cov_rank) / in_features)

    # 返回权重分布
    def W(self):
        cov_diag = torch.exp(self.W_logdiag)
        if self.W_dist == Normal:
            cov = self.W_dist(self.W_mean, cov_diag)
        elif (self.W_dist == DenseNormal) or (self.W_dist == DenseNormalPrec):
            tril = torch.tril(self.W_offdiag, diagonal=-1) + torch.diag_embed(cov_diag)
            cov = self.W_dist(self.W_mean, tril)
        elif self.W_dist == LowRankNormal:
            cov = self.W_dist(self.W_mean, self.W_offdiag, cov_diag)

        return cov

    # 返回噪声分布
    def noise(self):
        return Normal(self.noise_mean, torch.exp(self.noise_logdiag))

    # 前向传播，返回预测值、训练和验证损失函数
    def forward(self, x):
        out = VBLLReturn(self.predictive(x),
                         self._get_train_loss_fn(x),
                         self._get_val_loss_fn(x))
        return out

    # 计算预测分布
    def predictive(self, x):
        return (self.W() @ x[..., None]).squeeze(-1) + self.noise()

    # 获取训练损失函数
    def _get_train_loss_fn(self, x):

        def loss_fn(y):
            # construct predictive density N(W @ phi, Sigma)
            W = self.W()
            noise = self.noise()
            pred_density = Normal((W.mean @ x[..., None]).squeeze(-1), noise.scale)
            pred_likelihood = pred_density.log_prob(y)

            trace_term = 0.5 * ((W.covariance_weighted_inner_prod(x.unsqueeze(-2)[..., None])) * noise.trace_precision)

            kl_term = gaussian_kl(W, self.prior_scale)
            wishart_term = (self.dof * noise.logdet_precision - 0.5 * self.wishart_scale * noise.trace_precision)
            total_elbo = torch.mean(pred_likelihood - trace_term)
            total_elbo += self.regularization_weight * (wishart_term - kl_term)
            return -total_elbo

        return loss_fn

    # 获取验证损失函数
    def _get_val_loss_fn(self, x):
        def loss_fn(y):
            # compute log likelihood under variational posterior via marginalization
            logprob = self.predictive(x).log_prob(y).sum(-1)  # sum over output dims
            return -logprob.mean(0)  # mean over batch dim

        return loss_fn


class tRegression(nn.Module):
    """
    Variational Bayesian Linear Student-t Regression

    This version of the VBLL regression layer also infers noise covariance.

    Parameters
    ----------
    in_features : int
        Number of input features
    out_features : int
        Number of output features
    regularization_weight : float
        Weight on regularization term in ELBO
    parameterization : str
        Parameterization of covariance matrix. Currently supports {'dense', 'diagonal', 'lowrank'}
    prior_scale : float
        Scale of prior covariance matrix
    wishart_scale : float
        Scale of Wishart prior on noise covariance
    dof : float
        Degrees of freedom of Wishart prior on noise covariance

    in_features : int
    输入特征数
    out_features : int
    输出特征数
    regularization_weight : float
    ELBO中的正则化项权重
    parameterization : str
    协方差矩阵的参数化方法，支持{'dense', 'diagonal', 'lowrank'}
    prior_scale : float
    先验协方差矩阵的尺度
    wishart_scale : float
    噪声协方差的Wishart先验尺度
    dof : float
    噪声协方差的Wishart
    """

    def __init__(self,
                 in_features,
                 out_features,
                 regularization_weight,
                 parameterization='dense',
                 prior_scale=100.,
                 wishart_scale=100.,
                 cov_rank=None,
                 dof=1.):
        super(tRegression, self).__init__()

        self.wishart_scale = wishart_scale
        self.regularization_weight = regularization_weight

        # define prior, currently fixing zero mean and arbitrarily scaled cov
        self.prior_dof = dof
        self.prior_rate = 1. / wishart_scale
        self.prior_scale = prior_scale * (2. / in_features)  # kaiming init

        # variational posterior over noise params
        self.noise_log_dof = nn.Parameter(torch.ones(out_features) * np.log(self.prior_dof))
        self.noise_log_rate = nn.Parameter(torch.ones(out_features) * np.log(self.prior_rate))

        # last layer distribution
        self.W_dist = get_parameterization(parameterization)
        self.W_mean = nn.Parameter(torch.randn(out_features, in_features))

        self.W_logdiag = nn.Parameter(torch.randn(out_features, in_features))
        if parameterization == 'dense':
            self.W_offdiag = nn.Parameter(torch.randn(out_features, in_features, in_features))
        elif parameterization == 'lowrank':
            self.W_offdiag = nn.Parameter(torch.randn(out_features, in_features, cov_rank))
        elif parameterization == 'dense_precision':
            raise NotImplementedError()

    @property
    def W(self):
        cov_diag = torch.exp(self.W_logdiag)
        if self.W_dist == Normal:
            cov = self.W_dist(self.W_mean, cov_diag)
        elif self.W_dist == DenseNormal:
            tril = torch.tril(self.W_offdiag, diagonal=-1) + torch.diag_embed(cov_diag)
            cov = self.W_dist(self.W_mean, tril)
        elif self.W_dist == LowRankNormal:
            cov = self.W_dist(self.W_mean, self.W_offdiag, cov_diag)

        return cov

    @property
    def noise(self):
        noise_dof = torch.exp(self.noise_log_dof)
        noise_rate = torch.exp(self.noise_log_rate)
        return torch.distributions.gamma.Gamma(noise_dof, noise_rate)

    @property
    def noise_prior(self):
        return torch.distributions.gamma.Gamma(self.prior_dof, self.prior_rate)

    def forward(self, x):
        out = VBLLReturn(self.predictive(x),
                         self._get_train_loss_fn(x),
                         self._get_val_loss_fn(x))
        return out

    def predictive(self, x):
        dof = 2 * self.noise.concentration
        Wx = (self.W @ x[..., None]).squeeze(-1)
        mean = Wx.mean
        pred_cov = (Wx.variance + 1) * self.noise.rate / self.noise.concentration
        return torch.distributions.studentT.StudentT(dof, mean, torch.sqrt(pred_cov))

    def _get_train_loss_fn(self, x):

        def loss_fn(y):
            cov_factor = self.noise.concentration / self.noise.rate
            pred_err = (y - (self.W.mean @ x[..., None]).squeeze(-1)) ** 2
            pred_likelihood = (cov_factor * pred_err).sum(-1)

            logdet_term = (torch.digamma(self.noise.concentration) - torch.log(self.noise.rate)).sum(-1)
            trace_term = (self.W.covariance_weighted_inner_prod(x.unsqueeze(-2)[..., None])).sum(-1)

            kl_term = expected_gaussian_kl(self.W, self.prior_scale, self.noise)
            kl_term += gamma_kl(self.noise, self.noise_prior)

            total_elbo = -0.5 * torch.mean(pred_likelihood + trace_term - logdet_term)
            total_elbo -= self.regularization_weight * kl_term
            return -total_elbo

        return loss_fn

    def _get_val_loss_fn(self, x):
        def loss_fn(y):
            # compute log likelihood under variational posterior via marginalization
            logprob = self.predictive(x).log_prob(y).sum(-1)  # sum over output dims
            return -logprob.mean(0)  # mean over batch dim

        return loss_fn
