import numpy as np
from dataset_utilities.transforms import Transform
from dataset_utilities.utils import gamma_correction
from . import InstantKey, Instant


class GammaCorrectionTransform(Transform):
    def __init__(self, transform_dict=None):
        self.transform_dict = {
            29582 : [1.04, 1.02, 0.93], # Gravelines game
            24651 : [1.05, 1.02, 0.92], # Gravelines game
            69244 : [1.035, 1.025, 0.990], # Gravelines game
            59201 : [1.040, 1.030, 0.990], # Gravelines game
            30046 : [0.98, 0.98, 0.98], # Strasbourg game
            # TODO: LAPUA
            # TODO: ESPOO
            **(transform_dict if transform_dict is not None else {})
        }
        # transform_dict is a dict of game_id, gamma correction triplets
        assert all([isinstance(k, int) for k in self.transform_dict.keys()])

    def __call__(self, instant_key: InstantKey, instant: Instant):
        if instant_key.game_id in self.transform_dict.keys():
            gammas = np.array(self.transform_dict[instant_key.game_id])
            for k, image in instant.all_images.items():
                instant.all_images[k] = gamma_correction(image, gammas)
        return instant
