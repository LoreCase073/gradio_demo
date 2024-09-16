import abc
from decord import VideoReader, cpu
from utils.general_utils import sample_indices, write_subtitles
import os
import numpy as np
import av


class BaseMultimodalMethod(metaclass=abc.ABCMeta):

    def __init__(self, model_type: str, device, num_frames: int, frame_rate: int, output_path: str, experiment_name: str, video_folder:str, model_savepath: str = '', gradio_usage: bool = False):
        """
        Initializes the BaseMethod class.
        Args:
            model_type (str): The type of the model.
            device: The device to be used for computation.
            num_frames (int): The number of frames in the video.
            frame_rate (int): The frame rate of the video.
            output_path (str): The path to the output directory.
            experiment_name (str): The name of the experiment.
            video_folder (str): The folder containing the video files.
            model_savepath (str, optional): The path to save the model. Defaults to ''.
            gradio_usage (bool, optional): Whether Gradio is being used. Defaults to False.
        """
        self.model_type = model_type
        self.device_type = device
        self.num_frames = num_frames
        self.frame_rate = frame_rate
        # if the model needs to be saved in a directory different 
        # from the standard huggingface choice of .cache
        self.model_savepath = model_savepath
        self.base_output_path = output_path
        self.experiment_name = experiment_name
        self.video_folder = video_folder
        
        if not gradio_usage:
            self.base_csv_path = os.path.join(output_path,'csv_captions')
            if not os.path.exists(self.base_csv_path):
                os.mkdir(self.base_csv_path)
            self.base_srt_path = os.path.join(output_path,'srt_captions')
            if not os.path.exists(self.base_srt_path):
                os.mkdir(self.base_srt_path)
            self.csv_path = os.path.join(self.base_csv_path, self.experiment_name)
            if not os.path.exists(self.csv_path):
                os.mkdir(self.csv_path)
            self.csv_path = os.path.join(self.csv_path, self.video_folder)
            if not os.path.exists(self.csv_path):
                os.mkdir(self.csv_path)
            self.srt_path = os.path.join(self.base_srt_path, self.experiment_name)
            if not os.path.exists(self.srt_path):
                os.mkdir(self.srt_path)
            self.srt_path = os.path.join(self.srt_path, self.video_folder)
            if not os.path.exists(self.srt_path):
                os.mkdir(self.srt_path)
        


    @abc.abstractmethod
    def inference(self, *args):
      pass 


    @abc.abstractmethod
    def caption_video_file(self, *args):
      pass 


    def extract_indices(self, video_path: str):
        videoreader = VideoReader(video_path, num_threads=1, ctx=cpu(0))
        indices = sample_indices(self.num_frames, videoreader, self.frame_rate)
        del videoreader
        return indices
    

    def extract_frames(self, video_path: str, indices):
        videoreader = VideoReader(video_path, num_threads=1, ctx=cpu(0))
        video_frames = videoreader.get_batch(indices).asnumpy()
        return video_frames
    

    def read_video_pyav(self, video_path: str, indices):
        '''
        Decode the video with PyAV decoder.
        Args:
            container (`av.container.input.InputContainer`): PyAV container.
            indices (`List[int]`): List of frame indices to decode.
        Returns:
            result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
        '''
        container = av.open(video_path)
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i >= start_index and i in indices:
                frames.append(frame)
        return np.stack([x.to_ndarray(format="rgb24") for x in frames])
    

    def generate_subtitles_file(self, video_path: str, file_name: str, text_prompt: str):
        """
        Writes subtitles on a video file.
        Args:
            video_path (str): The path to the video file.
            file_name (str): The name of the video file.
            text_prompt (str): The text to be written as subtitles.
        Returns:
            None
        """
        csv_name = 'output_{}.csv'.format(file_name.replace('.mp4',''))
        srt_name = 'output_{}.srt'.format(file_name.replace('.mp4',''))
        csv_path_file = os.path.join(self.csv_path,csv_name)
        srt_path_file = os.path.join(self.srt_path,srt_name)

        
        write_subtitles(video_path, csv_path_file, srt_path_file, text_prompt)
        
        