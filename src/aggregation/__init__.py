from .fedavg import FedAvgAggregator
from .clip import ClipAggregator
from .deepsight import DeepSightAggregator
from .foolsgold import FoolsGoldAggregator
from .rflbat import RFLBATAggregator
from .apra import APRAAggregator

def get_aggregator(agg_method, helper):
    aggregators = {
        'avg': FedAvgAggregator,
        'clip': ClipAggregator,
        'deepsight': DeepSightAggregator,
        'foolsgold': FoolsGoldAggregator,
        'rflbat': RFLBATAggregator,
        'apra': APRAAggregator,
    }
    if agg_method not in aggregators:
        raise ValueError(f"Unknown aggregation method: {agg_method}. "
                         f"Available: {list(aggregators.keys())}")
    return aggregators[agg_method](helper)
