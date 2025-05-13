# pip install simpleaudio
# 터미널 실행 프로프트 : python 0.test_sounds.py 
import simpleaudio as sa

filename = 'assets/alarm.wav'
wave_obj = sa.WaveObject.from_wave_file(filename)
play_obj = wave_obj.play()
play_obj.wait_done()