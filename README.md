# byopVC
# VideoCrux: Summarizing Educational Videos

VideoCrux is a tool designed to assist learners by generating concise summaries of long educational videos. It extracts key points from video content, allowing users to revisit the main ideas without rewatching the entire video. By leveraging advanced AI models like OpenAI Whisper and T5, VideoCrux provides an efficient way to simplify note-taking and enhance comprehension.

Features

Video-to-Summary Pipeline: Converts video content into concise textual summaries.
High Accuracy: Uses fine-tuned T5 and Whisper models for transcription and summarization.
Real-World Application: Designed for summarizing lectures, tutorials, and other educational videos.

# File Descriptions

t5trainscicum.py: Script for fine-tuning the T5 model on the SciSum dataset.

textsum.py: Fine-tunes the T5 model on the CNN/Daily Mail dataset.

vcbyop.py: The main script that combines all components to generate summaries.

video downloader.py: Downloads videos from YouTube using the yt-dlp library.

vidtoaud.py: Extracts audio from video files using the ffmpeg library.

vidtotext.py: Combines audio extraction and transcription using Whisper.

# How It Works

Video Download: Videos are downloaded using the video downloader.py script.
Audio Extraction: The vidtoaud.py script extracts audio from the downloaded video.
Transcription: The extracted audio is transcribed into text using Whisper in the vidtotext.py script.
Summarization: Transcripts are broken into chunks of 512 tokens and summarized using the fine-tuned T5 model.
Cleanup: Intermediate audio and video files are deleted to save storage space.

# Key Results

Fine-Tuned T5 Model:
Dataset Used: CNN/Daily Mail (300,000 articles) and SciSum (1,000 research papers).
# ROUGE Scores:

ROUGE-1: 20.21
ROUGE-2: 4.82
ROUGE-L: 15.50

Comparison with Baseline:
Baseline ROUGE-1: ~15%
Fine-tuned ROUGE-1: 20.21% (significant improvement).


# Usage

Run the main script vcbyop.py to process a video:

python vcbyop.py --video_url <video_url>

# Inputs

A YouTube video link.

# Outputs

A concise summary of the video.




 
