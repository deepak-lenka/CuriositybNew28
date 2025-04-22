# MysteryBot - Anachronism Challenge

MysteryBot is an interactive web application that generates images containing anachronisms or impossible elements using the Grok3 image generation API. Users are challenged to identify what's out of place in each image.

## Features

- Generates images with anachronisms using the Grok3 API
- Presents multiple choice options for identifying the anachronism
- Provides explanations for each mystery
- Allows users to download the generated images
- Responsive web interface with modern design

## Setup Instructions

### Prerequisites

- Python 3.7+
- Grok3 API key (from X platform)

### Installation

1. Clone this repository or download the files

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Create a `.env` file in the project root directory with your Grok3 API key:
```
GROK_API_KEY=your_api_key_here
```

### Running the Application

1. Start the Flask web server:
```
python app.py
```

2. Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

## How It Works

1. The application selects a random prompt describing a scene with an anachronism (something out of its time period) or an impossible element
2. It uses the Grok3 API to generate an image based on the prompt
3. The user is presented with multiple choice options to identify what's out of place
4. After selecting an answer, the user receives feedback and an explanation

## Using the MysteryBot Class

You can also use the `MysteryBot` class directly in your Python code:

```python
from image import MysteryBot

# Initialize with your API key
bot = MysteryBot(api_key="your_api_key_here")

# Generate a mystery
mystery = bot.create_mystery()

if mystery["success"]:
    print(f"Mystery prompt: {mystery['prompt']}")
    print(f"Image URL: {mystery['image_url']}")
    print("Options:")
    for i, option in enumerate(mystery['options']):
        print(f"{i+1}. {option}")
else:
    print(f"Error: {mystery['error']}")
```

## Adding New Mystery Prompts

You can add new mystery prompts by editing the `MYSTERY_PROMPTS` list in `image.py`. Each prompt should include:

- `prompt`: The description for generating the image
- `anachronism`: The out-of-place element
- `explanation`: An explanation of why the element is anachronistic

## Notes

- The Grok3 Image Generation API (grok-2-image-1212) does not support the 'size' parameter
- The download functionality uses JavaScript to force the browser to download the image instead of opening it in a new tab
