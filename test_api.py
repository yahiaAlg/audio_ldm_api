import requests
import base64
import soundfile as sf
import io
import time


def test_local_api(prompt):
    url = "http://localhost:8000/generate-audio"

    payload = {
        "prompt": prompt,
        "audio_length": 5.0,
        "num_inference_steps": 10,
        "guidance_scale": 2.5,
    }

    print(f"Sending request for prompt: '{prompt}'")
    start_time = time.time()

    try:
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            end_time = time.time()
            print(
                f"Generation successful! Time taken: {end_time - start_time:.2f} seconds"
            )

            data = response.json()
            audio_base64 = data["audio_base64"]
            audio_data = base64.b64decode(audio_base64)

            # Save to file
            filename = f"generated_audio_{int(time.time())}.wav"
            with open(filename, "wb") as f:
                f.write(audio_data)
            print(f"Audio saved to {filename}")

            # Load audio data
            audio_io = io.BytesIO(audio_data)
            audio_array, sample_rate = sf.read(audio_io)
            return audio_array, sample_rate
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None, None
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None, None


# Test the API
prompt = "Techno music with a strong, upbeat tempo and high melodic riffs"
audio_array, sample_rate = test_local_api(prompt)
