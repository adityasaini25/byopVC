import yt_dlp
import ffmpeg
import whisper


video_url = input('enter video url: ' )
output_path = "./Data/video.mp4"

ydl_opts = {
    'format': 'best',
    'outtmpl': output_path,
    'noplaylist': True,
}
try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print(f"Downloading video from: {video_url}")
        ydl.download([video_url])
        print(f"Video downloaded successfully to: {output_path}")
except Exception as e:
    print(f"An error occurred: {e}")



input_video = './Data/video.mp4'
output_audio = './Data/audio.wav'
ffmpeg.input(input_video).output(output_audio).run()



from pydub import AudioSegment


audio = AudioSegment.from_file('./Data/audio.wav')


chunk_length_ms = 30 * 1000  # 30 seconds
chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]


for idx, chunk in enumerate(chunks):
    chunk.export(f"chunk_{idx}.wav", format="wav")




model = whisper.load_model("base")


full_transcription = []
for idx in range(len(chunks)):
    audio_path = f"chunk_{idx}.wav"
    result = model.transcribe(audio_path, task="transcribe")
    full_transcription.append(result["text"])

# Combine transcriptions
final_transcription = " ".join(full_transcription)
print("Final Transcription:", final_transcription)

