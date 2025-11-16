"""
Simple test script to validate ElevenLabs TTS from this workspace.
Usage (PowerShell):
  $env:ELEVENLABS_API_KEY = "sk_..."
  python .\test_elevenlabs_tts.py

This script lists available voices, picks one (matching VOICE_NAME if set),
requests TTS for a short sample text and writes `assets/test_welcome.mp3`.
"""
import os
import requests

API_KEY = os.environ.get("ELEVENLABS_API_KEY")
VOICE_NAME = "Bella"

if not API_KEY:
    print("Set ELEVENLABS_API_KEY in environment before running this test.")
    raise SystemExit(1)

def list_voices():
    url = "https://api.elevenlabs.io/v1/voices"
    r = requests.get(url, headers={"xi-api-key": API_KEY}, timeout=10)
    print("List voices status:", r.status_code)
    try:
        data = r.json()
    except Exception:
        print("Failed to parse voices response:\n", r.text)
        return []
    voices = data.get("voices") if isinstance(data, dict) else data
    if not voices:
        print("No voices returned")
        return []
    for v in voices[:20]:
        print("-", v.get("name") or v.get("voice_name") or v.get("label"), "-> id:", v.get("voice_id") or v.get("id"))
    return voices

voices = list_voices()
voice_id = None
for v in voices:
    name = v.get("name") or v.get("voice_name") or v.get("label")
    vid = v.get("voice_id") or v.get("id")
    if name and vid and name.lower() == VOICE_NAME.lower():
        voice_id = vid
        break
if not voice_id and voices:
    voice_id = voices[0].get("voice_id") or voices[0].get("id")

if not voice_id:
    print("Could not resolve a voice id to use. Aborting.")
    raise SystemExit(1)

print("Using voice id:", voice_id)

text = "Welcome to our site. Please select a monument to find out more about it."
url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
headers = {"xi-api-key": API_KEY}
payload = {"text": text, "voice_settings": {"stability": 0.5, "similarity_boost": 0.7}}

print("Requesting TTS...")
resp = requests.post(url, headers=headers, json=payload, timeout=30)
print("TTS status:", resp.status_code)
if resp.status_code == 200:
    os.makedirs("assets", exist_ok=True)
    path = os.path.join("assets", "test_welcome.mp3")
    with open(path, "wb") as f:
        f.write(resp.content)
    print("Saved TTS to", path)
else:
    print("TTS failed:\n", resp.status_code, resp.text)
