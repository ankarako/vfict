from typing import Dict, Any
import tyro
from ha.state.track import TrackState
from ha.util.conf import read_conf

def main(conf: str) -> None:
    """
    Main function wrapper
    """
    conf = read_conf(conf)
    state = TrackState(**conf)


    

if __name__ == "__main__":
    tyro.extras.set_accent_color("bright_yellow")
    cfg = tyro.cli(main)
    main(cfg)
    