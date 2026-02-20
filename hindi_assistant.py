import queue
import sounddevice as sd
import json
import vosk
import os
from llama_cpp import Llama

VOSK_MODEL = "vosk-model-small-hi-0.22"
LLM_MODEL = "/home/pi/llama.cpp/model.gguf"

vosk_model = vosk.Model(VOSK_MODEL)

llm = Llama(
    model_path=LLM_MODEL,
    n_ctx=256,
    n_threads=4
)

samplerate = 16000
q = queue.Queue()

def speak(text):
    print("Assistant:", text)
    os.system(f'espeak -v hi "{text}"')

def callback(indata, frames, time, status):
    q.put(bytes(indata))

print("Offline Hindi AI Assistant Ready")

with sd.RawInputStream(
        samplerate=samplerate,
        blocksize=8000,
        dtype='int16',
        channels=1,
        callback=callback):

    rec = vosk.KaldiRecognizer(vosk_model, samplerate)

    while True:
        data = q.get()

        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text = result.get("text", "")

            if text == "":
                continue

            print("You:", text)

            if "बंद" in text:
                speak("अलविदा")
                break

            output = llm(
                f"User: {text}\nAssistant:",
                max_tokens=80
            )

            response = output["choices"][0]["text"].strip()

            if response == "":
                response = "मुझे समझ नहीं आया"

            speak(response)