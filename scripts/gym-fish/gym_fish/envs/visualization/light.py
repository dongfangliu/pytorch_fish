from typing import Tuple

class pointLight:
    def __init__(self,pos:Tuple[float],color:Tuple[float]=(1.0, 1.0, 1.0, 0.25)) -> None:
        self.pos = pos
        self.color = color
