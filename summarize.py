from transformers import pipeline
import pandas as pd

summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)


df = pd.read_csv('lvm_data/results/csv_captions/new_llava_video_try_1/Coda/output_fuoco_2_000_030.csv')
captions = df['Caption'].tolist()
to_remove = "USER: \nWhat is happening on this highway, in this video? Point out if there are slowdowns or if the traffic is heavy. ASSISTANT: "
#TODO: tagliare prompt in maniera più intelligente.
processed_captions = []
for cap in captions:
    processed_captions.append(cap.replace(to_remove,''))


long_single_caption = " ".join(processed_captions)

length = int(len(long_single_caption)/2)


print(summarizer(long_single_caption[:4000], max_length=130, min_length=30, do_sample=False))