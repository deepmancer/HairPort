import os
import time
from PIL import Image
from google import genai
from google.genai import types

# --- CONFIGURATION ---
API_KEY = "AIzaSyB-mRtNoH60Ib6te31aXdBcwrKvuy6uFMs"  # Replace with your actual API Key
INPUT_FOLDER = "/workspace/outputs/image/"      # Directory containing your source PNGs
OUTPUT_FOLDER = "/workspace/outputs/bald_images_gemeni/"       # Directory where processed images will be saved
MODEL_ID = "gemini-3-pro-image-preview" # This is the "Nano Banana Pro" model

# The prompt instructions for the edit
EDIT_PROMPT = (
    "Edit this image to remove all hair from the person's head, making them completely bald. "
    "Preserve the person's facial features, identity, lighting, and the original background exactly. "
    "Do not change the art style."
)

to_replace_balds = [
    "ana_2",
    "bale_1", #  nano
    "eren_1", # nano
    "gojo_2", # nano
    "john_snow_1", # nano
    "joker_2", # nano
    "sample_002",
    "sample_041", # nano
    "sample_042",
    "sample_047",
    "sample_048",
    "sample_051",
    "sample_059",
    "sample_066",
    "sample_070",
    "sample_076",
    "sample_086",
    "sample_089",
    "huntington",
    "mave",
    "pedro_3",
    "rick_2",
    "sample_095",
    "sample_102",
    "sample_106",
    "sample_110",
    "sample_118",
    "sample_122",
    "sample_128",
    "sample_133",
    "sample_135",
    "sample_136",
    "sample_138",
    "sample_140",
    "sample_146",
    "sample_164",
    "sample_166",
    "sample_167",
    "sample_168",
    "sample_169",
    "sample_171",
    "superman",
    "taylor_1",
    "taylor_3",
    "thor_2",
    "vanessa_2",
    "vanessa_3",
    "vanessa",
    "wolvorine_1",
]


def process_images():
    # 1. Initialize the Client
    client = genai.Client(api_key=API_KEY)

    # 2. Ensure output directory exists
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # 3. List all PNG files
    files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith('.png')]
    total_files = len(files)
    
    print(f"Found {total_files} images to process in '{INPUT_FOLDER}'.")

    for index, filename in enumerate(files):
        file_id, _ = os.path.splitext(filename)
        if file_id not in to_replace_balds:
            print(f"[{index+1}/{total_files}] Skipping {filename} (not in to_replace_balds)")
            continue
        input_path = os.path.join(INPUT_FOLDER, filename)
        output_path = os.path.join(OUTPUT_FOLDER, filename)

        # Skip if already processed (optional, good for resuming)
        if os.path.exists(output_path):
            print(f"[{index+1}/{total_files}] Skipping {filename} (already exists)")
            continue

        print(f"[{index+1}/{total_files}] Processing {filename}...")

        try:
            # 4. Load the image as bytes
            with open(input_path, "rb") as f:
                image_bytes = f.read()
            
            # Create an image part with proper MIME type
            image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")

            # 5. Call the API
            # We pass the image bytes along with the text prompt.
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=[image_part, EDIT_PROMPT],
                config=types.GenerateContentConfig(
                    response_modalities=["IMAGE"], # Force image output
                )
            )

            # 6. Extract and Save the Result
            # The response contains the generated image data
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.inline_data:
                        # as_image() returns a PIL Image directly
                        generated_image = part.as_image()
                        generated_image.save(output_path)
                        print(f"   Success! Saved to {output_path}")
                        break
            else:
                print(f"   Failed: No image returned for {filename}")

        except Exception as e:
            print(f"   Error processing {filename}: {e}")
            # Optional: Sleep briefly to avoid hitting rate limits too hard if errors occur
            time.sleep(1)

        # Rate Limiting: "Nano Banana Pro" is a large model. 
        # Sleep slightly between requests to be safe (adjust based on your tier limit).
        time.sleep(2) 

if __name__ == "__main__":
    process_images()