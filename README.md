# 🎵 Music AI — Harta Interactivă

An interactive web application showcasing **Romanian monuments** with AI-generated music, trivia, and text-to-speech descriptions.  

Supports **map-based interaction**, **Solana donations**, and integrates multiple AI services to enhance the experience.

---

## Features

### 🗺 Interactive Map
- Click on a static map of **Romania** or **Timișoara** to select monuments.
- Preselects nearby monuments automatically.
- Toggle between **Romania** and **Timișoara** views.
- Shows markers and coordinates of monuments.

### 🎶 AI Music Generation
- Generates music for the selected monument using **HuggingFace models**.
- **Gemini AI** improves prompts sent to the music generator for better contextual relevance.
- Plays generated music directly in the browser.
- **Note:** Music generation can take a few seconds. To keep users engaged, a separate **C#/.NET mini-games site** provides games like **Rock-Paper-Scissors**, **trivia questions**, and more.

### 🧠 Trivia & Description Generation
- Generates trivia questions for selected monuments using **Gemini AI**.
- Provides descriptive text for each monument.
- Optional **Text-to-Speech** using **ElevenLabs API**.

### 💜 Donations
- Integrated **Solana Pay** for donations.
- Works with Phantom, Solflare, Backpack, or any Solana wallet.
- QR code fallback for users without wallet extensions.

---

## Tech Stack

- **Frontend:** Blazor WebAssembly + Gradio for interactive Python components
- **Backend:** ASP.NET Core API + Python AI scripts
- **Music Generation:** HuggingFace models + Gemini prompt enhancement
- **Trivia Generation:** Gemini AI
- **Text-to-Speech:** ElevenLabs API
- **Blockchain Payments:** Solana Pay
- **Mini-Games:** C# + .NET (RPS, trivia, etc.)
- **Maps:** Static images with coordinate detection
- **AI Components:** Music generation, trivia, TTS, and prompt optimization

---

## Getting Started

### Prerequisites

- [.NET 8 SDK](https://dotnet.microsoft.com/en-us/download/dotnet/8.0)
- Python 3.10+ with required packages (`gradio`, `requests`, `openai`, `huggingface_hub`, etc.)
- Solana wallet (Phantom or other)
- ElevenLabs API key

## Getting Started

### Prerequisites

- [.NET 8 SDK](https://dotnet.microsoft.com/en-us/download/dotnet/8.0)
- Python 3.10+ with required packages (`gradio`, `requests`, `openai`, `huggingface_hub`, etc.)
- Solana wallet (Phantom or other)
- ElevenLabs API key

### Installation

1. Clone the repository:

```bash
git clone https://github.com/andreeaonaci/Unihack_Monument_History_AI.git
cd monument-music-ai
```

2. Set the environment variables: AUTH0_AUDIENCE, AUTH0_CLIENT_ID, AUTH0_CLIENT_SECRET, AUTH0_DOMAIN, ELEVENLABS_API_KEY, GEMINI_API_KEY.
