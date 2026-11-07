
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Step 1: Load API key
print("Loading API key...", flush=True)
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Gemini API key not found in .env!")

# Step 2: Initialize client with a 20-second timeout
print("Connecting to Gemini...", flush=True)

client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(timeout=20000)
)

# Step 3: Send request
print("Sending request to Gemini...", flush=True)

try:
    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents="What is Artificial Intelligence? Explain in simple words."
    )

    print("\nLLM Response:")
    print(response.text)

except Exception as e:
    print("\nAPI REQUEST FAILED")
    print("Error type:", type(e).__name__)
    print("Error details:", str(e))

finally:
    client.close()
    print("\nProgram finished.")
