from typing import Optional
import imageio
import numpy as np

class VideoWriter:
    def __init__(self, path: str, fps: int = 60):
        self.path = path
        self.fps = int(fps)
        self._writer = imageio.get_writer(path, fps=self.fps, codec="libx264", quality=8)

    def add(self, frame_rgb: np.ndarray):
        self._writer.append_data(frame_rgb)

    def close(self):
        self._writer.close()
