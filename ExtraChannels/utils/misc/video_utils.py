import os
import torch
import numpy as np
import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from tqdm import tqdm
from .preprocess_texture import preprocess_style_image, preprocess_video, RGBToEdges

os.environ['FFMPEG_BINARY'] = 'ffmpeg'


class VideoWriter:
    def __init__(self, filename='tmp.mp4', fps=30.0, autoplay=False, **kw):
        self.writer = None
        self.autoplay = autoplay
        self.params = dict(filename=filename, fps=fps, **kw)

    def add(self, img):
        img = np.asarray(img)
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
        if img.dtype in [np.float32, np.float64]:
            img = np.uint8(img.clip(0, 1) * 255)
        if len(img.shape) == 2:
            img = np.repeat(img[..., None], 3, -1)
        self.writer.write_frame(img)

    def close(self):
        if self.writer:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *kw):
        self.close()
        if self.autoplay:
            self.show()

    def show(self, **kw):
        self.close()
        fn = self.params['filename']
        display(mvp.ipython_display(fn, **kw))



def save_video(video_name, target_vid_path, size_factor=1.0, step_n=8, steps_per_frame=1, is_style_image=False, nca_model=None, nca_size_x=256, nca_size_y=256, DEVICE='cuda', autoplay=True):
    target_vid = preprocess_video(target_vid_path,
                                    img_size=(int(nca_size_x * size_factor), int(nca_size_y * size_factor)))  # [C, T, H, W]


    target_vid = target_vid.permute(1,0,2,3).to(DEVICE)

    with VideoWriter(filename=f"{video_name}.mp4", fps=30, autoplay=autoplay) as vid, torch.no_grad():
        h = nca_model.seed(1, size=(int(nca_size_x * size_factor), int(nca_size_y * size_factor)))



        for frame in tqdm(range(target_vid.size(0)),  desc="Making the video..."):
            for k in range(int(steps_per_frame)):
                f = 1 if not is_style_image else 90
                for i in range(f):

                    h = torch.cat((h, target_vid[frame].unsqueeze(0)), 1)
                    nca_state, nca_feature = nca_model.forward_nsteps(h, step_n)

                    z = nca_feature
                    h = nca_state[:, :-3, :, :]

                    img = z.detach().cpu().numpy()[0]
                    img = img.transpose(1, 2, 0)
                    img = np.clip(img, -1.0, 1.0)
                    img = (img + 1.0) / 2.0
                    vid.add(img)
