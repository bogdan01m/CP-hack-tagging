from huggingface_hub import hf_hub_download
import av
import numpy as np
from read_and_play import read_video_pyav

# Download video from the hub
video_path_1 = hf_hub_download(repo_id="raushan-testing-hf/videos-test", filename="sample_demo_1.mp4", repo_type="dataset")

container = av.open(video_path_1)

# sample uniformly 8 frames from the video (we can sample more for longer videos)
total_frames = container.streams.video[0].frames
indices = np.arange(0, total_frames, total_frames / 8).astype(int)
clip_baby = read_video_pyav(container, indices)