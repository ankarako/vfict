from typing import Dict, Any
import t3d
import tloaders

class TrackState:
    def __init__(
        self,
        ict_model_kwargs: Dict[str, Any],
        dataset_conf: Dict[str, Any]
    ) -> None:
        """
        Tracking state. Holds the state for tracking videos
        """
        self.ict_model = t3d.mms.ICTModel(**ict_model_kwargs)
        self.dataset = tloaders.DatasetRegistry.get_dataset(
            dataset_conf.key, **dataset_conf.kwargs
        )

        self.optim = None

def configure_params(state: TrackState) -> None:
    """
    """