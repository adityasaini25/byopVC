import ffmpeg


input_video = './Data/video.mp4'


output_audio = './Data/audio.wav'


ffmpeg.input(input_video).output(output_audio).run()
