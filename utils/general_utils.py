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

    video_len = len(videoreader)
    # sample 6 frames
    videoreader.seek(0)
    indices = sample_frame_indices(clip_len=num_frames, frame_sample_rate=frame_rate, video_len=video_len)
 

    return indices



def get_values_based_on_index(df, index):
    # Check if index is between 'Start' and 'End' for each row
    mask = (index >= df['Start']) & (index <= df['End'])
    values = df.loc[mask, 'Caption'].tolist()
    if not values:
        previous_row_index = (df['Start'] < index).idxmax() - 1  # Find the index of the previous row
        if previous_row_index >= 0:
            return [df.loc[previous_row_index, 'Caption']]
    return values


def scale_text_properties(width, height, base_font_scale=1, base_thickness=2):
    scale = min(width, height) / 1080  # Assuming 1080p as base resolution
    font_scale = base_font_scale * scale
    thickness = int(base_thickness * scale)
    return font_scale, thickness


def frame_to_timecode(frame, fps):
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





#----------------------- OUTDATED: this is the functioning version -------------------

""" def write_text_on_video(file_path, csv_path, output_path, font=cv2.FONT_HERSHEY_SIMPLEX,
                               font_scale=1, font_color=(255, 255, 255), font_thickness=2, line_spacing=20):

    video = cv2.VideoCapture(file_path)
    csv = pd.read_csv(csv_path)
    # Get video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Get scaled font properties
    font_scale, font_thickness = scale_text_properties(width, height)

    
    frame_number = 0
    
    print('Adding text to video.')
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        
        # Check if the current frame number is in the CSV data
        text = get_values_based_on_index(csv, frame_number)
        if text != [] and text != ['_']:
            text = text[0]

            # Add text to the frame
            
            # Calculate the size of the text
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)

            # Split text into multiple lines if needed
            max_width = width - 20  # Leave some padding
            wrapped_text = textwrap.wrap(text, width=max_width//(cv2.getTextSize(" ", font, font_scale, font_thickness)[0][0] + 1))

            # Calculate the starting y position
            line_height = text_size[1] + line_spacing
            total_text_height = line_height * len(wrapped_text)
            start_y = height - total_text_height - 10  # 10 pixels from the bottom
            
            # Add each line of text to the frame
            for i, line in enumerate(wrapped_text):
                text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
                text_x = (width - text_size[0]) // 2
                text_y = start_y + i * line_height
                cv2.putText(frame, line, (text_x, text_y), font, font_scale, font_color, font_thickness)



        
        # Write the frame into the output video
        out.write(frame)
        
        frame_number += 1
    
    # Release everything if the job is finished
    video.release()
    out.release()
    cv2.destroyAllWindows() """


#--------------- OUTDATED: possible changes, too slow however-------------------
""" 
def add_text_with_pillow(frame, text, position, text_color, border_color, font_size):
    # Convert the frame to a PIL image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    
    # Load the custom font
    if font_size < 1:
        font_size = 1
    font = ImageFont.truetype("./lvm_mov/methods/font_file/OpenSans-Bold.ttf", font_size)

    # Calculate border thickness proportionate to the font size
    border_thickness = max(1, int(font_size / 10))

    x, y = position
    # Draw border by drawing the text multiple times around the position
    for dx in range(-border_thickness, border_thickness + 1):
        for dy in range(-border_thickness, border_thickness + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=border_color)
    
    # Draw the actual text on top
    draw.text(position, text, font=font, fill=text_color)
    
    # Convert the PIL image back to an OpenCV format
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)



def write_text_on_video(file_path, csv_path, output_path, font=cv2.FONT_HERSHEY_SIMPLEX,
                               font_scale=20, font_color=(255, 255, 255), border_color=(0,0,0), font_thickness=2, line_spacing=20):

    video = cv2.VideoCapture(file_path)
    csv = pd.read_csv(csv_path)
    # Get video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Get scaled font properties
    #font_scale, font_thickness = scale_text_properties(width, height)

    
    frame_number = 0
    
    print('Adding text to video.')
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        print('Frame number: {}'.format(frame_number))
        # Check if the current frame number is in the CSV data
        text = get_values_based_on_index(csv, frame_number)
        if text != [] and text != ['_']:
            text = text[0]

            # Add text to the frame
            
            # Calculate the size of the text
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)

            # Split text into multiple lines if needed
            max_width = width - 20  # Leave some padding
            wrapped_text = textwrap.wrap(text, width=max_width//(cv2.getTextSize(" ", font, font_scale, font_thickness)[0][0] + 1))

            # Calculate the starting y position
            line_height = text_size[1] + line_spacing
            total_text_height = line_height * len(wrapped_text)
            start_y = height - total_text_height - 10  # 10 pixels from the bottom
            
            # Add each line of text to the frame
            for i, line in enumerate(wrapped_text):
                text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
                text_x = (width - text_size[0]) // 2
                text_y = start_y + i * line_height
                #cv2.putText(frame, line, (text_x, text_y), font, font_scale, font_color, font_thickness)
                frame = add_text_with_pillow(frame, line, position=(text_x, text_y), text_color=font_color, border_color=border_color, font_size=font_scale)



        
        # Write the frame into the output video
        out.write(frame)
        
        frame_number += 1
    
    # Release everything if the job is finished
    video.release()
    out.release()
    cv2.destroyAllWindows() """