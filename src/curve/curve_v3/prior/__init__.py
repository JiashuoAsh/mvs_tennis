"""curve_v3 第一阶段（prior）相关模块。

说明：
    - 这里放与“反弹前 prefit、候选生成、先验模型、在线沉淀”相关的逻辑。
    - 目的是减少 `curve_v3/` 根目录下文件数量，让结构更好读。
"""

from curve_v3.prior.candidates import build_prior_candidates
from curve_v3.prior.models import PriorModel, PriorFeatures, RbfSamplePriorModel, UniformPriorModel, features_from_v_minus
from curve_v3.prior.online_integration import maybe_init_online_prior, maybe_update_online_prior
from curve_v3.prior.online_prior import OnlinePriorWeights, load_or_create_online_prior
from curve_v3.prior.prefit import estimate_bounce_event_from_prefit

__all__ = [
    "PriorFeatures",
    "PriorModel",
    "RbfSamplePriorModel",
    "UniformPriorModel",
    "OnlinePriorWeights",
    "build_prior_candidates",
    "estimate_bounce_event_from_prefit",
    "features_from_v_minus",
    "load_or_create_online_prior",
    "maybe_init_online_prior",
    "maybe_update_online_prior",
]
