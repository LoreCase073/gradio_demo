import gradio as gr
from methods.llavanextvideo import LlavaNextVideo
from methods.summarizer import Summarizer
import subprocess
import os
import argparse


method_name = 'llava_next_video'
device_id = 0
exp_name = 'gradio'
video_folder = ''

model = LlavaNextVideo(model_type=method_name, device=device_id, model_savepath=None, output_path=None, experiment_name=exp_name, video_folder=video_folder, gradio_usage=True)
summarizer = Summarizer('',exp_name, gradio_usage=True)

def get_length(filename):
    """
    Get the length of a video file.

    Parameters:
    filename (str): The path to the video file.

    Returns:
    float: The duration of the video file in seconds.
    """
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)


def convert_to_mp4(video_path):
    """
    Converts a video file to the mp4 format.

    Args:
        video_path (str): The path to the video file.

    Returns:
        str: The path to the converted mp4 file, or an empty string if the conversion failed.
    """
    file_name, file_extension = os.path.splitext(video_path)
    if file_extension != ".mp4":
        print("Converting to mp4.")
        new_file_path = f"{file_name}.mp4"
        result = subprocess.run(["ffmpeg", "-i", video_path, "-qscale", "0", new_file_path, "-y"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        if result.returncode == 0 and os.path.exists(new_file_path):
            print(f"Conversion successful! File saved as {new_file_path}")
        else:
            print("Conversion failed.")
            new_file_path = ''
            return new_file_path
            # Optionally, print the error message
            #print(result.stdout.decode('utf-8'))
    else:
        new_file_path = video_path
    return new_file_path

def cut_long_video(video_path):
    """
    Cuts a long video to a specified duration.

    Args:
        video_path (str): The path to the input video file.

    Returns:
        str: The path to the cut video file.

    Raises:
        subprocess.CalledProcessError: If the ffmpeg command fails to execute.

    """
    # Rest of the code...
    print("Cutting video because too long!")
    file_name, file_extension = os.path.splitext(video_path)
    cut_video_path = f"{file_name}_cut.mp4"
    result = subprocess.run(["ffmpeg", "-ss", "00:00:00", "-t", "00:00:10", "-i",
                             video_path, "-c", "copy", cut_video_path, "-y"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
    return cut_video_path

def predict(message, history, last_video):
    """
    Predicts the captions and summary for a given video and user input.

    Parameters:
    - message (dict): A dictionary containing the user input and video path.
                      Format: {"text": "user input", "files": ["file_path1", "file_path2", ...]}
    - history (list): A list containing the history of user inputs and generated summaries.
    - last_video (str): The path of the last video that was processed.

    Returns:
    - gr.MultimodalTextbox: A Gradio MultimodalTextbox component.
    - history (list): The updated history list.
    - gr.ClearButton: A Gradio ClearButton component.
    """
    ...
    # for multimodal: {"text": "user input", "files": ["file_path1", "file_path2", ...]}
    if last_video != '':
        video_path = last_video
        prompt = [
                    {

                        "role": "user",
                        "content": [
                            {"type": "text", "text": "{}".format(message['text'])},
                            {"type": "video"},
                            ],
                    },
                ]
        captions, csv_path_file = model.gradio_video_inference(video_path,prompt,max_length = 100)
        summary = summarizer.gradio_summarizer(captions)
        history.append((None,summary))
        history.append((None,'You can download the results of the inference clicking on the link below.'))
        history.append((None,gr.File(csv_path_file)))
        return gr.MultimodalTextbox(value=None, interactive=True),history, gr.ClearButton(components=[history, last_video], interactive=True)
    else:
        history.append((None,'Load a video to start the inference!'))
        return gr.MultimodalTextbox(value=None, interactive=True),history, gr.ClearButton(components=[history, last_video], interactive=True)
    

def new_multimodal_message(message,last_video,chatbot, clear_btn):
    """
    Process a new multimodal message.
    Args:
        message (dict): The message containing text and files.
        last_video (str): The path of the last video.
        chatbot (list): The list of chatbot messages.
        clear_btn (gr.ClearButton): The clear button component.
    Returns:
        tuple: A tuple containing the updated components:
            - gr.MultimodalTextbox: The updated multimodal textbox component.
            - str: The updated path of the last video.
            - list: The updated list of chatbot messages.
            - gr.ClearButton: The updated clear button component.
    """
    chatbot.append((message['text'],None))
    if len(message['files']) == 1:
        last_video = message['files'][0]

        last_video = convert_to_mp4(last_video)
        # if last_video = '' there was an error in the conversion
        if last_video != '':
            video_duration = get_length(last_video)
            
            if int(video_duration) >= 11:
                video_too_long_response = 'Video is too long! Only the first 10 seconds of video will be analyzed. Limit to 10 seconds next time.\n\n'
                last_video = cut_long_video(last_video)
                chatbot.append((gr.Video(last_video),None))
                chatbot.append((None,video_too_long_response))
                return gr.MultimodalTextbox(interactive=False),last_video, chatbot, gr.ClearButton(components=[chatbot, last_video], interactive=False)
            elif int(video_duration) < 2:
                # do not save a video too short
                chatbot.append((None,'Video is too short. Input a video of at least 2 seconds.'))
                last_video = ''
                return gr.MultimodalTextbox(interactive=False),last_video, chatbot, gr.ClearButton(components=[chatbot, last_video], interactive=True)
            else:
                chatbot.append((gr.Video(last_video),None))
                return gr.MultimodalTextbox(interactive=False),last_video,chatbot, gr.ClearButton(components=[chatbot, last_video], interactive=False)
        else:
            # do not save a video too short
            chatbot.append((None,'Error during the conversion.'))
            last_video = ''
            return gr.MultimodalTextbox(interactive=False),last_video, chatbot, gr.ClearButton(components=[chatbot, last_video], interactive=True)
    return gr.MultimodalTextbox(interactive=False),last_video, chatbot, gr.ClearButton(components=[chatbot, last_video], interactive=False)



def main(user, password):
    with gr.Blocks(fill_height=True) as demo:
        last_video = gr.State('')

        
        chatbot = gr.Chatbot(height="70vh")
        clear_btn = gr.ClearButton(components=[chatbot, last_video], render=True)
        msg = gr.MultimodalTextbox(interactive=True)
        chat_msg = msg.submit(new_multimodal_message,[msg,last_video,chatbot, clear_btn],[msg,last_video, chatbot, clear_btn])
        bot_msg = chat_msg.then(predict,inputs=[msg, chatbot, last_video], outputs=[msg,chatbot, clear_btn])

    # msg = gr.MultimodalTextbox(interactive=True)
    # msg.submit(predict,inputs=[msg,chatbot,last_video],outputs=[msg,chatbot,last_video])
    #gr.ChatInterface(predict, multimodal=True, additional_inputs=[last_video], chatbot=chatbot,textbox=msg, retry_btn=None, undo_btn= None, clear_btn=clear_btn)
    if user != '' and password != '':
        demo.launch(share=True, auth=(user,password))
    else:
        print("User or password were not provided. Launching without sharing.")
        demo.launch()


if __name__ == "__main__":
    description = 'TBD'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--user', type=str, default='',
                   help=('Username to use in authentication'))
    p.add_argument('--password', type=str, default='',
                   help=('Password to use in authentication.'))
    
    args = p.parse_args()

    user = args.user
    password = args.password
    main(user,password)