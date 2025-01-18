from transformers import T5Tokenizer, T5ForConditionalGeneration

from vidtotext import final_transcription
import os

model_name = "./t5-small-finetuned-dailym/checkpoint-17945"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

input_text = ("convert to passive:  " + final_transcription)
chunk_size = 512

chunks = [input_text[i:i + chunk_size] for i in range(0, len(input_text), chunk_size)]
converted_chunks = []




input_text = ("summarize:  " + input_text)
chunk_size = 512


chunks = [input_text[i:i + chunk_size] for i in range(0, len(input_text), chunk_size)]
summaries = []




for chunk in chunks:

   inputs = tokenizer(
       chunk,
       return_tensors="pt",
       truncation=True,
       max_length=512,
       padding="max_length"
   )



   outputs = model.generate(
       input_ids=inputs["input_ids"],
       attention_mask=inputs["attention_mask"],
       max_length=100,
       do_sample=True,
       top_k=50,
       top_p=0.9,
       temperature=1.0,

   )



   summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
   summaries.append(summary)
full_summary = " ".join(summaries)
print(full_summary)

audio_file = "./Data/audio.wav"
video_file = "./Data/video.mp4"
try:
    if os.path.exists(audio_file):
        os.remove(audio_file)
        print(f"Deleted {audio_file}")
    else:
        print(f"{audio_file} does not exist.")

    if os.path.exists(video_file):
        os.remove(video_file)
        print(f"Deleted {video_file}")
    else:
        print(f"{video_file} does not exist.")
except Exception as e:
    print(f"Error deleting files: {e}")





