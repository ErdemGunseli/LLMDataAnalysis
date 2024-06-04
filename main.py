from typing import List
import base64
import os

from dotenv import load_dotenv
import requests
import pandas as pd
from openai import OpenAI
from pydub import AudioSegment
import simpleaudio as sa

from strings import *


"""
An API-based approach has the following benefits over interacting with the Web UI manually:

1)  Full Automation - The code and prompts can be executed several times, whereas using the UI requires repeated inputs.

2)  Reusability - The code and prompts can be written once and reused across various datasets, 
    whereas this is requires repeated inputs using the Web UI.

3)  System Instructions - Using the API allows us to set the 'role' field of certain messages to be 'system'. 
    This has two benefits: 
    a. These instructions fine-tune behavior and context of the assistant directly.
    b.  System instructions are less likely to be lost through the smart context window of the LLM, 
    meaning they are retained more consistently compared to normal prompts sent through the Web UI.

4)  Data Security - Requests made to the API do not have the risk of being unintentionally exposed,
    as the content is excluded from the training data by default. 

5)  Better Dataset Transfer - Due to token limits for requests, the dataset needs to be broken up into chunks. 
    Using the Web UI, this is a slow and manual process, whereas with the API, this process can be automated, 
    ensuring more efficient handling of larger datasets.
"""

client = OpenAI()


def text_to_speech(text: str) -> str: 
    response = client.audio.speech.create(model="tts-1", voice="alloy", input=text)
    audio_content = response.content
    # Base64 encoding the audio:
    return base64.b64encode(audio_content).decode("utf-8")


def play_audio(encoded_audio):
    print("Playing audio...")

    # Decoding the base64 string:
    audio_bytes = base64.b64decode(encoded_audio)
    
    # Saving the audio to a file:
    with open("output.wav", "wb") as f:
        f.write(audio_bytes)
    
    # Loading the audio file
    audio = AudioSegment.from_file("output.wav")
    
    # Playing the audio:
    play_obj = sa.play_buffer(audio.raw_data, num_channels=audio.channels, bytes_per_sample=audio.sample_width, sample_rate=audio.frame_rate)
    
    # Waiting for playback to finish before exiting:
    play_obj.wait_done()


def send_messages(messages) -> List[str]:

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}

    # Sending a request with the messages:
    payload = {"model": "gpt-4", "messages": messages, "max_tokens": 400}
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    # Ensuring response is successful:
    if response.status_code != 200: 
        raise Exception(f"Request failed with status code {response.status_code}: {response.text}")

    # Extracting the response content:
    response_data = response.json()
    response_text = response_data["choices"][0]["message"]["content"]

    # Adding ChatGPT's response to the messages:
    messages.append({"role": "assistant", "content": response_text})

    return messages


def read_csv(path="Churn.csv", subset=100, chunk_length=50) -> List[str]:
    # Reading the CSV file:
    df = pd.read_csv(path)
    
    # Including the desired subset amount:
    df_subset = df.head(subset)
    
    # Converting the CSV content to a string:
    csv_content = df_subset.to_string(index=False)
    
    # Breaking the content into chunks:
    chunks = [csv_content[i:i+chunk_length] for i in range(0, len(csv_content), chunk_length)]

    return chunks



def main():
    load_dotenv()

    csv_chunks = read_csv()

    messages = [{"role": "system", "content": TASK_DESCRIPTION}, {"role": "system", "content": CONTEXT}]

    # Adding the CSV dataset chunks:
    messages += [{"role": "user", "content": chunk} for chunk in csv_chunks] 
    
    response = send_messages(messages)[-1].get("content")

    print(f"Response from GPT4.o: {response}")

    audio_output = text_to_speech(response)

    play_audio(audio_output)



if __name__ == "__main__": main()
    

"""
python main.py
"""