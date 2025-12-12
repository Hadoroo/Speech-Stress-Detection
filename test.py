import soundfile as sf
import sounddevice as sd

file = "LDC99S78.1 (1).sph"

data, samplerate = sf.read(file)
sd.play(data, samplerate)
sd.wait()