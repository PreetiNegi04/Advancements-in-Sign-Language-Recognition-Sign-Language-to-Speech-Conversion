print("Testing print statements")

import pyttsx3

def play_sound(text):
    print("Initializing TTS engine...")
    engine = pyttsx3.init()
    print("Setting voice...")
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)
    print(f"Speaking: {text}")
    engine.say(text)
    engine.runAndWait()
    print("Speech completed.")

play_sound("This is a test.")
