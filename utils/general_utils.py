import numpy as np
from itertools import cycle, islice
import pandas as pd
from moviepy.editor import VideoFileClip



def sample_frame_indices(clip_len, frame_sample_rate, video_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample per clip.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): length of the video.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    indices_list = []
    count_clip = 0
    indices = []
    for i in range(video_len):
        if i % frame_sample_rate == 0:
            indices.append(i)
            count_clip += 1
        if count_clip == clip_len:
            indices_list.append(np.array(indices).astype(np.int64))
            indices.clear()
            count_clip = 0

    if len(indices_list) == 0:
        indices_list.append(np.array(indices).astype(np.int64))
    
    if len(indices_list[-1]) < clip_len:
        indices_list[-1] = np.array(list(islice(cycle(indices_list[-1]),clip_len))).astype(np.int64)

    return indices_list


def sample_indices(num_frames, videoreader, frame_rate):
    """
    Sample a set of frame indices from a video.
    Args:
        num_frames (int): The number of frames to sample.
        videoreader (object): An object that allows reading frames from a video.
        frame_rate (int): The rate at which frames should be sampled.
    Returns:
        list: A list of sampled frame indices.
    """
    video_len = len(videoreader)
    # sample 6 frames
    videoreader.seek(0)
    indices = sample_frame_indices(clip_len=num_frames, frame_sample_rate=frame_rate, video_len=video_len)
 

    return indices




def frame_to_timecode(frame, fps):
    """
    Converts a given frame number to a timecode string in the format HH:MM:SS,mmm.
    Args:
        frame (int): The frame number to convert.
        fps (float): The frames per second (fps) of the video.
    Returns:
        str: The timecode string in the format HH:MM:SS,mmm.
    """
    seconds = frame / fps
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{int(seconds):02},{milliseconds:03}"

def create_srt_from_csv_with_frames(csv_path, srt_path, fps, text_prompt):
    """
    Create an SRT file from a CSV file containing frame information.

    Args:
        csv_path (str): The path to the CSV file.
        srt_path (str): The path to the output SRT file.
        fps (float): The frames per second of the video.
        text_prompt (str): The text prompt to remove from captions.

    Returns:
        None
    """
    df = pd.read_csv(csv_path)

    with open(srt_path, 'w') as f:
        for index, row in df.iterrows():
            start_frame = row['Start']
            end_frame = row['End']
            text = row['Caption'].replace('\n', '')
            text = row['Caption'].replace(str(text_prompt), '')

            start_time = frame_to_timecode(start_frame, fps)
            end_time = frame_to_timecode(end_frame, fps)

            f.write(f"{index + 1}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n\n")



def write_subtitles(file_path, csv_path, srt_path, text_prompt: str):
    """
    Writes subtitles for a video file based on a CSV file.

    Args:
        file_path (str): The path to the video file.
        csv_path (str): The path to the CSV file containing subtitle data.
        srt_path (str): The path to save the generated SRT file.
        text_prompt (str): The text prompt to be used for generating subtitles.

    Returns:
        None
    """

    video = VideoFileClip(file_path)
    # Get video properties
    fps = video.fps
    create_srt_from_csv_with_frames(csv_path, srt_path, fps=fps, text_prompt=text_prompt)



