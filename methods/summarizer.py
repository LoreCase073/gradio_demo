from transformers import pipeline
import pandas as pd
import abc
import os

class Summarizer(metaclass=abc.ABCMeta):

    def __init__(self, output_path, experiment_name, gradio_usage: bool = False):
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)
        self.base_output_path = output_path
        self.experiment_name = experiment_name
        if not gradio_usage:
            self.base_summary_path = os.path.join(output_path,'summary')
            if not os.path.exists(self.base_summary_path):
                os.mkdir(self.base_summary_path)
            self.summary_path = os.path.join(self.base_summary_path, self.experiment_name)
            if not os.path.exists(self.summary_path):
                os.mkdir(self.summary_path)



    def split_into_chunks(self, text, max_tokens=1024, min_last_chunk_tokens=130):
        words = text.split()
        chunks = [' '.join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]

        # Check if the last chunk has fewer tokens than min_last_chunk_tokens
        if len(chunks) > 1 and len(chunks[-1].split()) < min_last_chunk_tokens:
            # Concatenate the second last and last chunk
            chunks[-2] = f"{chunks[-2]} {chunks[-1]}"
            chunks.pop()  # Remove the last chunk after concatenation
        
        return chunks

    


    def summarize_to_csv(self, video_name, csv_path, prompt_to_delete):
        df = pd.read_csv(csv_path)
        captions = df['Caption'].tolist()
        processed_captions = []
        for cap in captions:
            processed_captions.append(cap.replace(prompt_to_delete,''))

        long_single_caption = " ".join(processed_captions)
        caption_chunks = self.split_into_chunks(long_single_caption, max_tokens=512, min_last_chunk_tokens=130)
        summary_chunks = []
        for i, chunk in enumerate(caption_chunks):
            chunk_token_length = len(chunk.split())
            if chunk_token_length >= 130:
                summary_out = self.summarizer(chunk, max_length=130, min_length=30, do_sample=False)
                summary_chunks.append(summary_out[0]['summary_text'])
            else:
                summary_chunks.append(chunk)
        
        summary_tmp = " ".join(summary_chunks)
        summary_length = len(summary_tmp.split())
        if summary_length >= 130:
            summary_final = self.summarizer(summary_tmp, max_length=130, min_length=30, do_sample=False)
        else:
            summary_final = [summary_tmp]
        data_dict = {'Summary': summary_final}
            
        #write_csv
        df = pd.DataFrame(data_dict)
        summary_name = 'summary_{}.csv'.format(video_name.replace('.mp4',''))
        df.to_csv(os.path.join(self.summary_path,summary_name), index=False, header=True)

    def gradio_summarizer(self, prompt_to_sumamrize):
        long_single_caption = " ".join(prompt_to_sumamrize)
        summary_out = self.summarizer(long_single_caption, max_length=130, min_length=30, do_sample=False)
        summary_out = summary_out[0]
        summary_out = summary_out['summary_text']
        return summary_out
        
