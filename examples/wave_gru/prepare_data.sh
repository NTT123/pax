# "Yoshua Bengio: Deep Learning Cognition | Full Keynote - AI in 2020 & Beyond"
youtube-dl -f 139 https://www.youtube.com/watch?v=GibjI5FoZsE --output /tmp/wave_gru_clip.m4a
# convert m4a to wav
ffmpeg -i /tmp/wave_gru_clip.m4a -ac 1 -ar 16000 -acodec pcm_s16le /tmp/wave_gru_clip_.wav
# trim silences
sox /tmp/wave_gru_clip_.wav /tmp/wave_gru_clip.wav silence -l 1 0.1 1% -1 1.0 1%
