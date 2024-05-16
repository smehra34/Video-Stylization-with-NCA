import os
import torch
import numpy as np
import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from tqdm import tqdm
from .preprocess_texture import preprocess_style_image, preprocess_video, RGBToEdges
from PIL import Image


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



def save_video(video_name, target_vid_path, size_factor=1.0, step_n=8,
               steps_per_frame=1, is_style_image=False, nca_model=None,
               nca_size_x=256, nca_size_y=256, DEVICE='cuda', autoplay=True):

    if not is_style_image:
        target_vid = preprocess_video(target_vid_path,
                                      img_size=(int(nca_size_x * size_factor),
                                                int(nca_size_y * size_factor)))  # [C, T, H, W]
    else:
        target_vid = video_like_process_style_image(target_vid_path,
                                                    img_size=(int(nca_size_x * size_factor),
                                                              int(nca_size_y * size_factor)))


    target_vid = target_vid.permute(1,0,2,3).to(DEVICE)

    with VideoWriter(filename=f"{video_name}.mp4", fps=30, autoplay=autoplay) as vid, torch.no_grad():
        h = nca_model.seed(1, size=(int(nca_size_x * size_factor), int(nca_size_y * size_factor)))



        for frame in tqdm(range(target_vid.size(0)),  desc="Making the video..."):
            for k in range(int(steps_per_frame)):

                h = torch.cat((h, target_vid[frame].unsqueeze(0)), 1)
                nca_state, nca_feature = nca_model.forward_nsteps(h, step_n)

                z = nca_feature
                h = nca_state[:, :-3, :, :]

                img = z.detach().cpu().numpy()[0]
                img = img.transpose(1, 2, 0)
                img = np.clip(img, -1.0, 1.0)
                img = (img + 1.0) / 2.0
                vid.add(img)


def evaluate_folder_of_videos(video_dir, save_path, size_factor=1.0, step_n=8,
                              steps_per_frame=1, nca_model=None, nca_size_x=256,
                              nca_size_y=256, DEVICE='cuda'):

    target_videos = [f for f in os.listdir(video_dir) if f.split('.')[-1] in ['mp4', 'gif']]
    target_video_paths = [os.path.join(video_dir, v) for v in target_videos]
    target_vid_names = [f.split('.')[0] for f in target_videos]

    print(f"Evaluating on the following {len(target_videos)} videos: {', '.join(target_videos)}")

    for i, vid_path in enumerate(target_video_paths):
        print(target_vid_names[i])
        save_video(f"{save_path}/{target_vid_names[i]}", target_vid_path=vid_path, size_factor=size_factor,
                    step_n=step_n, steps_per_frame=steps_per_frame, nca_model=nca_model,
                    nca_size_x=nca_size_x, nca_size_y=nca_size_y, DEVICE=DEVICE, autoplay=False)

def generate_control_videos(style_img_path, save_path, size_factor=1.0, step_n=8,
                              steps_per_frame=1, nca_model=None, nca_size_x=256,
                              nca_size_y=256, DEVICE='cuda'):

    print('Generating control videos...')

    style_img = Image.open(style_img_path)
    black_img = Image.new('RGB', (nca_size_x, nca_size_y), color='black')

    print('Style image')
    save_video(f"{save_path}/style_img", target_vid_path=style_img, size_factor=size_factor,
                step_n=step_n, steps_per_frame=steps_per_frame, is_style_image=True,
                nca_model=nca_model, nca_size_x=nca_size_x, nca_size_y=nca_size_y,
                DEVICE=DEVICE, autoplay=False)

    print('Black image')
    save_video(f"{save_path}/black_img", target_vid_path=black_img, size_factor=size_factor,
                step_n=step_n, steps_per_frame=steps_per_frame, is_style_image=True,
                nca_model=nca_model, nca_size_x=nca_size_x, nca_size_y=nca_size_y,
                DEVICE=DEVICE, autoplay=False)


def video_like_process_style_image(style_image, img_size=[128, 128], normalRGB = False):

    img_tensor = preprocess_style_image(style_image, 'vgg', img_size)
    if(normalRGB == False):
        img_tensor = img_tensor * 2.0 - 1.0

    img_tensors = [img_tensor] * 250
    stack = torch.stack(img_tensors, dim=2)[0] # Output shape is [C, T, H, W]
    # print(f'Total Training Frames: {index}')
    return stack
