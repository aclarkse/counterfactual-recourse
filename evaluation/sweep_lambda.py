"""Compatibility entry point for the corrected independent recourse ablation.

The former implementation varied the shortfall weight while calling it lambda
and reported a pointwise direct gap as NIE_post. It has been removed. Use
``evaluation.ablate_recourse`` directly in new scripts.
"""

import hydra

from evaluation.ablate_recourse import run


@hydra.main(config_path="../conf", config_name="config", version_base="1.1")
def main(cfg):
    run(cfg)


if __name__ == "__main__":
    main()
