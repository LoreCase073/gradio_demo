import argparse
from methods.llavanextvideo import LlavaNextVideo
from methods.xclip import XCLIP
from methods.summarizer import Summarizer


def main(file_path: str, method_name: str, file_name: str, out_path: str, exp_name: str, video_folder:str,
         input_prompt: str, max_length: int = 100, no_creation: bool=False, image_in: bool=False, device_id: int=0,
         num_frames: int=10, frame_rate: int=3) -> None:

    if method_name == 'llava_next_video':
        approach = LlavaNextVideo(model_type=method_name, device=device_id, model_savepath='', 
                                  output_path=out_path, experiment_name=exp_name, video_folder=video_folder, 
                                  num_frames=num_frames, frame_rate=frame_rate)
        prompt = [
                    {

                        "role": "user",
                        "content": [
                            {"type": "text", "text": "{}".format(input_prompt)},
                            {"type": "video"},
                            ],
                    },
                ]
    elif method_name == 'xclip':
        approach = XCLIP(model_type=method_name, device=device_id, model_savepath='', 
                         output_path=out_path, experiment_name=exp_name, video_folder=video_folder, 
                         num_frames=num_frames, frame_rate=frame_rate)
        prompt = ['stopped vehicle', 'normal traffic']
    else:
        print('This method is not yet implemented')
        return 
    
    if image_in:
        approach.caption_image_file(file_path=file_path, image_name=file_name, prompt=prompt, no_creation=no_creation, max_length=max_length)
    else:
        csv_path_file, prompt_to_delete = approach.caption_video_file(file_path=file_path, video_name=file_name, prompt=prompt, no_creation=no_creation, max_length=max_length)
        if method_name != 'xclip':
            # summarize the video output
            summarizer = Summarizer(out_path,exp_name)
            summarizer.summarize_to_csv(file_name,csv_path_file,prompt_to_delete)


if __name__ == "__main__":
    description = 'TBD'
    p = argparse.ArgumentParser(description=description)
    p.add_argument('file_path', type=str,
                   help=('Path to folder'))
    p.add_argument('file_name', type=str,
                   help=('Name of the video.'))
    p.add_argument('method_name', type=str,
                   help=('Name of the method to be applied'))
    p.add_argument('out_path', type=str,
                   help=('Path where to save the video.'))
    p.add_argument('exp_name', type=str,
                   help=('name of the experiment.'))
    p.add_argument('video_folder', type=str,
                   help=('name of the video folder.'))
    p.add_argument('--max_length', type=int, default=100,
                   help=('Prompt to be fed to the method.'))
    p.add_argument('--prompt', type=str,
                   help=('Prompt to be fed to the method.'))
    p.add_argument('--no_creation', type=str, choices=['yes', 'no'],
                   help=('NoCreation.'))
    p.add_argument('--image', type=str, choices=['yes', 'no'], default='no',
                   help=('If to prompt an image.'))
    p.add_argument('--num_frames', type=int, default=10,
                   help=('How many frames to do inference with.'))
    p.add_argument('--frame_rate', type=int, default=3,
                   help=('Frame rate to samples the video.'))
    
    args = p.parse_args()

    if args.no_creation == 'yes':
        no_creation = True
    else:
        no_creation = False

    if args.image == 'yes':
        image_in = True
    else:
        image_in = False


    main(file_path=args.file_path, method_name=args.method_name, file_name=args.file_name, out_path=args.out_path, 
         exp_name=args.exp_name, video_folder=args.video_folder, input_prompt=args.prompt,
         max_length=args.max_length, no_creation=no_creation, image_in=image_in, device_id=0,
         num_frames=args.num_frames, frame_rate=args.frame_rate)