from mbrs import registry

from .base import Metric

register, get_cls = registry.setup("metric")

from .bleu import MetricBLEU
from .chrf import MetricChrF
from .comet import MetricCOMET

__all__ = ["Metric", "MetricBLEU", "MetricChrF", "MetricCOMET"]
