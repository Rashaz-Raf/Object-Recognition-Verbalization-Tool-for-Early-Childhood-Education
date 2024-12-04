from PIL import ImageGrab, Image
from gtts import gTTS
from faster_whisper import WhisperModel
import speech_recognition as sr
import google.generativeai as genai
import pyperclip
import cv2
import os
import time
import playsound


# Initialize APIs
wake_word = 'Jarvis'
genai.configure(api_key='AIzaSyBMNZ5Pi1j3nHsW2_om1Kvc-qdrsokHt5Q')

# Initialize webcam
web_cam = cv2.VideoCapture(1)

# System message for AI
sys_msg = (
    'You are a multi-modal AI voice assistant. Your user may or may not have attached a photo for context. '
    '(either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed '
    'text prompt that will be attached to their transcribed voice prompt. Generate the most useful and '
    'factual response possible, carefully considering all previous generated text in your response before '
    'adding new tokens to the response. Do not expect or request images, just use the context if added. '
    'Use all of the context of this conversation so your response is relevant to the conversation. Make '
    'your responses clear and concise, avoiding any verbosity.'
)

convo = [{'role': 'system', 'content': sys_msg}]
generation_config = {
    'temperature': 0.7,
    'top_p': 1,
    'top_k': 1,
    'max_output_tokens': 2048
}

safety_settings = [
    {'category': 'HARM_CATEGORY_HARASSMENT', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_HATE_SPEECH', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'threshold': 'BLOCK_NONE'},
    {'category': 'HARM_CATEGORY_DANGEROUS_CONTENT', 'threshold': 'BLOCK_NONE'},
]

model = genai.GenerativeModel(
    'gemini-1.5-flash-latest',
    generation_config=generation_config,
    safety_settings=safety_settings
)

num_cores = os.cpu_count()
whisper_size = 'base'

whisper_model = WhisperModel(
    whisper_size,
    device='cpu',
    compute_type='int8',
    cpu_threads=num_cores // 2,
    num_workers=num_cores // 2
)

r = sr.Recognizer()


def function_call(prompt):
    print(f'Calling function_call with prompt: {prompt}')
    sys_msg = (
        'You are an AI function calling model. You will determine whether extracting the user\'s clipboard content, '
        'taking a screenshot, capturing the webcam or calling no functions is best for a voice assistant to respond '
        'to the user\'s prompt. The webcam can be assumed to be a normal laptop webcam facing the user. You will '
        'respond with only one selection from this list: ["extract clipboard", "take screenshot", "capture webcam", "None"] \n'
        'Do not respond with anything but the most logical selection from that list with no explanations. Format the '
        'function call name exactly as I listed.'
    )

    function_convo = [{'role': 'system', 'content': sys_msg}, {'role': 'user', 'content': prompt}]
    chat_completion = model.generate_content(function_convo)
    response = chat_completion.text
    print(f'function_call response: {response}')
    return response

def take_screenshot():
    path = 'screenshot.jpg'
    screenshot = ImageGrab.grab()
    rgb_screenshot = screenshot.convert('RGB')
    rgb_screenshot.save(path, quality=15)
    print("Screenshot taken and saved as screenshot.jpg.")

def web_cam_capture():
    if not web_cam.isOpened():
        print('Error: Camera did not open successfully')
        return
    path = 'webcam.jpg'
    ret, frame = web_cam.read()
    if ret:
        cv2.imwrite(path, frame)
        print("Webcam captured and saved as webcam.jpg.")
    else:
        print("Error capturing webcam.")

def get_clipboard_text():
    clipboard_content = pyperclip.paste()
    if isinstance(clipboard_content, str) and clipboard_content.strip():
        print(f'Clipboard content retrieved: {clipboard_content}')
        return clipboard_content
    else:
        print('No clipboard text to copy')
        return None

def vision_prompt(prompt, photo_path):
    print(f'Calling vision_prompt with prompt: {prompt} and photo_path: {photo_path}')
    img = Image.open(photo_path)
    response = model.generate_content([prompt, img])
    print(f'vision_prompt response: {response.text}')
    return response.text


# Replaced Groq's speech system with gTTS for speaking responses
def speak(text):
    print(f'Speaking: {text}')
    tts = gTTS(text=text, lang='en')
    tts.save("response.mp3")
    playsound.playsound("response.mp3")
    os.remove("response.mp3")
    print('Speaking finished.')

    time.sleep(1)  # Add a pause of 1 second after speaking

def wav_to_text(audio_path):
    print(f'Converting audio to text from: {audio_path}')
    segments, _ = whisper_model.transcribe(audio_path)
    text = ''.join(segment.text for segment in segments)
    print(f'Transcribed text: {text}')
    return text

def extract_prompt(text, wake_word):
    if wake_word.lower() in text.lower():
        return text.lower().replace(wake_word.lower(), '').strip()
    return None

def callback(recognizer, audio):
    print('Voice detected, processing audio...')
    prompt_audio_path = 'prompt.wav'
    with open(prompt_audio_path, 'wb') as f:
        f.write(audio.get_wav_data())

    prompt_text = wav_to_text(prompt_audio_path)
    print(f'Transcribed Text: {prompt_text}')  # Debugging line
    clean_prompt = extract_prompt(prompt_text, wake_word)

    if clean_prompt:
        print(f'USER: {clean_prompt}')
        call = function_call(clean_prompt)
        print(f'Function Call: {call}')  # Debugging line

        if 'take screenshot' in call:
            print('Taking screenshot.')
            take_screenshot()
            visual_context = vision_prompt(prompt=clean_prompt, photo_path='screenshot.jpg')

        elif 'capture webcam' in call:
            print('Capturing webcam.')
            web_cam_capture()
            visual_context = vision_prompt(prompt=clean_prompt, photo_path='webcam.jpg')

        elif 'extract clipboard' in call:
            print('Extracting clipboard text.')
            paste = get_clipboard_text()
            if paste:
                clean_prompt = f'{clean_prompt} \n\n CLIPBOARD CONTENT: {paste}'
            visual_context = None

        else:
            visual_context = None

        response = model.generate_content([clean_prompt, visual_context])
        print(f'ASSISTANT: {response.text}')
        speak(response.text)
    else:
        print("No valid prompt detected.")

def start_listening():
    print("Starting to listen for wake word...")
    mic_list = sr.Microphone.list_microphone_names()
    print("Available Microphones:")
    for index, mic_name in enumerate(mic_list):
        print(f"{index}: {mic_name}")

    mic_index = 0  # Adjust index as necessary for your desired microphone

    print(f'\nUsing microphone: {mic_list[mic_index]} at index {mic_index}')

    with sr.Microphone(mic_index) as source:
        r.adjust_for_ambient_noise(source, duration=1)
        print("Listening...")

        while True:
            try:
                audio = r.listen(source, timeout=5)  # Timeout after 5 seconds
                callback(r, audio)
            except sr.WaitTimeoutError:
                print("Listening timed out; waiting for wake word again.")
            except Exception as e:
                print(f"An error occurred: {e}")

# Start the voice assistant
start_listening()
