from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration, BitsAndBytesConfig
import torch
import pandas as pd
import os
from tqdm import tqdm
from methods.base_method import BaseMultimodalMethod
from PIL import Image
import numpy as np

class LlavaNextVideo(BaseMultimodalMethod):

    
    def __init__(self, model_type: str, device, model_savepath: str, output_path: str, experiment_name: str, video_folder:str, num_frames: int = 10, frame_rate: int = 3, gradio_usage: bool = False):
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

        

         # specify how to quantize the model
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        # name of the model from huggingface
        model_name = "llava-hf/LLaVA-NeXT-Video-7B-DPO-hf"
        self.processor = LlavaNextVideoProcessor.from_pretrained(model_name, revision="00fe05a8e7214965362b0714e01a2536d85b3692")
        self.model = LlavaNextVideoForConditionalGeneration.from_pretrained(model_name, pad_token_id=self.processor.tokenizer.eos_token_id, quantization_config=quantization_config, device_map="auto")
        


    def get_frame_specifications(self):
        """
        Returns the number of frames and the frame rate.

        Returns:
            tuple: A tuple containing the number of frames and the frame rate.
        """
        return self.num_frames, self.frame_rate 
    

    def inference(self, frames, text_prompt: str, image_check: bool=False, max_length: int=100):
        """
        Perform inference using the LLavaNextVideo model.
        Args:
            frames (numpy.ndarray): The frames of the video as a numpy array.
            text_prompt (str): The text prompt for the model.
            image_check (bool, optional): Whether to use image input instead of video frames. Defaults to False.
            max_length (int, optional): The maximum length of the generated output. Defaults to 100.
        Returns:
            str: The generated output from the model.
        """
        frames = frames.astype(np.uint8)
        prompt = self.processor.apply_chat_template(text_prompt, add_generation_prompt=True)
        self.processor.tokenizer.padding_side = "left"
        
        if not image_check:
            if frames.shape[0] == self.num_frames:
                # prepare frames for the model
                inputs = self.processor(text=prompt, videos=frames, return_tensors="pt", padding=True).to("cuda")
                # forward pass
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=max_length)

                pred = self.processor.batch_decode(outputs,skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                return pred
            else:
                print('Skip this batch because not enough frames.')
                return "_"
        else:
            inputs = self.processor(text=prompt, images=frames, return_tensors="pt", padding=True).to("cuda")
            # forward pass
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=max_length)

            pred = self.processor.batch_decode(outputs,skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            return pred

    def caption_video_file(self, file_path: str, video_name: str, prompt: str, no_creation: bool, max_length: int = 100):
        """
        Caption a video file and generate a CSV file with captions.
        Args:
            file_path (str): The path to the directory containing the video file.
            video_name (str): The name of the video file.
            prompt (str): The text prompt to generate captions.
            no_creation (bool): If True, captions will not be generated.
            max_length (int, optional): The maximum length of the generated captions. Defaults to 100.
        Returns:
            Tuple[str, str]: A tuple containing the path to the generated CSV file and the modified text prompt.
        Raises:
            None
        """
        video_path = os.path.join(file_path, video_name)

        indices = self.extract_indices(video_path=video_path)
        csv_name = 'output_{}.csv'.format(video_name.replace('.mp4',''))
        csv_path_file = os.path.join(self.csv_path,csv_name)

        captions = []
        if not no_creation:

            print('Generating captions.')
            for idx in tqdm(range(len(indices))):
                frames = self.extract_frames(video_path=video_path, indices=indices[idx])
                if frames.shape[0] == self.num_frames:
                    out = self.inference(frames=frames, text_prompt=prompt, max_length=max_length)
                    captions.append(out)

            start_ind = [ind[0] for ind in indices]
            end_ind = [ind[-1] for ind in indices]

            data_dict = {'Start': start_ind,
                        'End': end_ind,
                        'Caption': captions}
            
            #write_csv
            df = pd.DataFrame(data_dict)
            df.to_csv(csv_path_file, index=False, header=True)

        # for now reencode always false, cause it depends on the ffmpeg installation of encoders
        text_prompt = self.processor.apply_chat_template(prompt, add_generation_prompt=True)
        text_prompt = text_prompt.replace('<video>','')
        self.generate_subtitles_file(video_path=video_path, file_name=video_name, text_prompt=text_prompt)
        return csv_path_file, text_prompt

    def gradio_video_inference(self, file_path: str, prompt: str, max_length: int = 100):
        """
        Perform video inference using Gradio.
        Args:
            file_path (str): The path to the video file.
            prompt (str): The prompt for generating captions.
            max_length (int, optional): The maximum length of the generated caption. Defaults to 100.
        Returns:
            tuple: A tuple containing the generated captions and the path to the CSV file where the captions are saved.
        """
        video_path = file_path
        tmp_dir = '/tmp/gradio'

        indices = self.extract_indices(video_path=video_path)
        text_prompt_to_delete = self.processor.apply_chat_template(prompt, add_generation_prompt=True)
        text_prompt_to_delete = text_prompt_to_delete.replace('<video>','')

        captions = []

        for idx in tqdm(range(len(indices))):
            frames = self.extract_frames(video_path=video_path, indices=indices[idx])
            #frames2 = self.read_video_pyav(video_path=video_path, indices=indices[idx])
            if frames.shape[0] == self.num_frames:
                out = self.inference(frames=frames, text_prompt=prompt, max_length=max_length)
                captions.append(out.replace(text_prompt_to_delete,''))

        csv_path_file = os.path.join(tmp_dir,'inferences.csv')

        start_ind = [ind[0] for ind in indices]
        end_ind = [ind[-1] for ind in indices]
        
        data_dict = {'Start': start_ind,
                        'End': end_ind,
                        'Caption': captions}
            
        #write_csv
        df = pd.DataFrame(data_dict)
        df.to_csv(csv_path_file, index=False, header=True)

        return captions, csv_path_file
        
        

    def caption_image_file(self, file_path: str, image_name: str, prompt: str, no_creation: bool, max_length: int = 100):
        """
        Caption an image file.
        Args:
            file_path (str): The path to the file.
            image_name (str): The name of the image file.
            prompt (str): The text prompt for generating captions.
            no_creation (bool): Flag indicating whether to skip caption generation.
            max_length (int, optional): The maximum length of the generated caption. Defaults to 100.
        """
        image_path = os.path.join(file_path, image_name)
        image = Image.open(image_path)
        csv_name = 'output_{}.csv'.format(image_name.replace('.png',''))
        csv_path_file = os.path.join(self.csv_path,csv_name)

        captions = []
        if not no_creation:

            print('Generating captions.')
            
            
            
            out = self.inference(frames=image, text_prompt=prompt, image_check=True, max_length=max_length)
            captions.append(out)
            print(out)