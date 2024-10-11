from transformers import AutoProcessor, AutoModel
import torch
import pandas as pd
import os
from tqdm import tqdm
from methods.base_method import BaseMultimodalMethod

class XCLIP(BaseMultimodalMethod):

    
    def __init__(self, model_type: str, device, model_savepath: str, output_path: str, experiment_name: str, video_folder:str, num_frames: int = 10, frame_rate: int = 30, gradio_usage: bool = False):
        """
        Initializes the LlavaNextVideo class.
        Args:
            model_type (str): The type of the model.
            device: The device to run the model on.
            model_savepath (str): The path to save the model.
            output_path (str): The output path for the generated videos.
            experiment_name (str): The name of the experiment.
            video_folder (str): The folder containing the videos.
            num_frames (int, optional): The number of frames to use for video generation. Defaults to 10.
            frame_rate (int, optional): The frame rate for the generated videos. Defaults to 3.
            gradio_usage (bool, optional): Whether to use Gradio for user interface. Defaults to False.
        """
        super().__init__(model_type, device, num_frames, frame_rate, output_path, experiment_name, video_folder, model_savepath, gradio_usage=gradio_usage)


        # name of the model from huggingface
        model_name = "microsoft/xclip-base-patch16-zero-shot"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda:{}".format(0) if torch.cuda.is_available() else "cpu"
        self.model.to(device)


    def get_frame_specifications(self):
        """
        Returns the number of frames and the frame rate.

        Returns:
            tuple: A tuple containing the number of frames and the frame rate.
        """
        return self.num_frames, self.frame_rate 
    

    def inference(self, frames, text_prompt: str, image_check: bool=False, max_length: int=100):
        """
        Perform inference on a batch of video frames with a given text prompt.

        Args:
            frames (torch.Tensor): A tensor containing video frames.
            text_prompt (str): The text prompt to be used for inference.
            image_check (bool, optional): If True, perform image check before inference. Defaults to False.
            max_length (int, optional): The maximum length for the text prompt. Defaults to 100.

        Returns:
            list or str: The probabilities of the text prompt matching the video frames if `image_check` is False and the number of frames is correct.
                         Returns "_" if the number of frames is not sufficient.
        """
        if not image_check:
            if frames.shape[0] == self.num_frames:
                # prepare frames for the model
                inputs = self.processor(text=text_prompt, videos=list(frames), return_tensors="pt", padding=True).to(self.device)
                # forward pass
                with torch.no_grad():
                    outputs = self.model(**inputs)

                probs = outputs.logits_per_video.softmax(dim=1).to('cpu').squeeze().tolist()
                return probs
            else:
                print('Skip this batch because not enough frames.')
                return "_"
        else:
            pass

    def caption_video_file(self, file_path: str, video_name: str, prompt: str, no_creation: bool, max_length: int = 100):
        """
        Generates captions for a video file and saves the results to a CSV file.
        Args:
            file_path (str): The directory path where the video file is located.
            video_name (str): The name of the video file.
            prompt (str): The text prompt to guide the caption generation.
            no_creation (bool): If True, skips the caption generation process.
            max_length (int, optional): The maximum length of the generated captions. Defaults to 100.
        Returns:
            tuple: A tuple containing the path to the CSV file and the generated captions.
        """
        video_path = os.path.join(file_path, video_name)

        indices = self.extract_indices(video_path=video_path)
        csv_name = 'output_{}.csv'.format(video_name.replace('.mp4',''))
        csv_path_file = os.path.join(self.csv_path,csv_name)

        if not no_creation:

            print('Generating captions.')
            for idx in tqdm(range(len(indices))):
                frames = self.extract_frames(video_path=video_path, indices=indices[idx])
                if frames.shape[0] == self.num_frames:
                    out = self.inference(frames=frames, text_prompt=prompt, max_length=max_length)


            data_dict = {'Prompt': prompt,
                        'Probs': out}
            
            #write_csv
            df = pd.DataFrame(data_dict)
            df.to_csv(csv_path_file, index=False, header=True)

        return csv_path_file, out

    def gradio_video_inference(self, file_path: str, prompt: str, max_length: int = 100):
        """
        Not Implemented
        """
        pass
        
        

    def caption_image_file(self, file_path: str, image_name: str, prompt: str, no_creation: bool, max_length: int = 100):
        """
        Not Implemented
        """
        pass