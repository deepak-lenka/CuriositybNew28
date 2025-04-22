import os
import random
import requests
import json
import base64
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Grok3 API configuration
GROK_API_KEY = os.getenv("GROK_API_KEY")
GROK_API_URL = "https://api.x.ai/v1/images/generations"
GROK_CHAT_API_URL = "https://api.x.ai/v1/chat/completions"

# Fallback mystery prompts (used only if API fails)
FALLBACK_MYSTERY_PROMPTS = [
    {
        "prompt": "A medieval knight holding a smartphone",
        "anachronism": "smartphone",
        "explanation": "Smartphones didn't exist in medieval times!"
    },
    {
        "prompt": "A dinosaur walking through a modern city",
        "anachronism": "dinosaur",
        "explanation": "Dinosaurs went extinct millions of years before modern cities existed!"
    },
    {
        "prompt": "A Victorian lady using a drone",
        "anachronism": "drone",
        "explanation": "Drones are modern technology that didn't exist in the Victorian era!"
    }
]

class MysteryBot:
    def __init__(self, api_key=None):
        """Initialize the MysteryBot with an optional API key"""
        self.api_key = api_key or GROK_API_KEY
        if not self.api_key:
            print("Warning: No Grok3 API key provided. Set the GROK_API_KEY environment variable.")
        
        # Model configuration
        self.image_model = "grok-2-image-1212"
        self.chat_model = "grok-3-beta"
        
        # Historical periods and modern items for generating anachronisms
        self.historical_periods = [
            "medieval times", "ancient Egypt", "Victorian era", "prehistoric age",
            "ancient Rome", "feudal Japan", "Wild West", "Renaissance period",
            "Stone Age", "ancient Greece", "Mayan civilization", "Viking age",
            "Byzantine Empire", "Ming Dynasty", "Ottoman Empire", "Edo period Japan"
        ]
        
        self.modern_items = [
            "smartphone", "laptop", "electric car", "solar panel", "robot vacuum",
            "digital camera", "smartwatch", "VR headset", "drone", "satellite dish",
            "microwave oven", "credit card", "LED light", "jet aircraft", "space shuttle",
            "television", "plastic bottle", "sneakers", "helicopter", "subway train"
        ]
    
    def generate_mystery_prompt(self):
        """Generate a new mystery prompt using Grok API"""
        if not self.api_key:
            return self.get_fallback_mystery()
        
        try:
            # First, randomly select a historical period and modern item
            period = random.choice(self.historical_periods)
            item = random.choice(self.modern_items)
            
            # Use Grok API to generate a more creative prompt and explanation
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            prompt = f"Create a mystery prompt for an image with an anachronism. The historical period is {period} and the anachronistic item is {item}. Return the result as JSON with these fields: prompt (the image generation prompt), anachronism (the out-of-place item), explanation (why it's anachronistic), and art_style (an interesting art style for the image)."
            
            messages = [
                {"role": "system", "content": "You are an expert at creating historical anachronism mysteries. You respond only with valid JSON objects."},
                {"role": "user", "content": prompt}
            ]
            
            payload = {
                "model": self.chat_model,
                "messages": messages,
                "response_format": {"type": "json_object"}
            }
            
            response = requests.post(GROK_CHAT_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            mystery_data = json.loads(content)
            
            # Enhance the prompt with the art style if provided
            if "art_style" in mystery_data and mystery_data["art_style"]:
                # Append the art style to the prompt
                mystery_data["prompt"] = f"{mystery_data['prompt']} in {mystery_data['art_style']} style"
            
            return mystery_data
            
        except Exception as e:
            print(f"Error generating mystery prompt: {str(e)}")
            return self.get_fallback_mystery()
    
    def get_fallback_mystery(self):
        """Get a random mystery prompt from the fallback list"""
        return random.choice(FALLBACK_MYSTERY_PROMPTS)
    
    def get_random_mystery(self):
        """Get a random mystery prompt, preferably generated dynamically"""
        return self.generate_mystery_prompt()
    
    def generate_options(self, correct_answer):
        """Generate multiple choice options for a mystery"""
        # Use the modern items list as the source of options
        incorrect_options = [item for item in self.modern_items if item != correct_answer]
        
        # If we don't have enough options, add some from the fallback prompts
        if len(incorrect_options) < 2:
            fallback_items = [item["anachronism"] for item in FALLBACK_MYSTERY_PROMPTS if item["anachronism"] != correct_answer]
            incorrect_options.extend(fallback_items)
        
        # Select 2 random incorrect options
        selected_incorrect = random.sample(incorrect_options, min(2, len(incorrect_options)))
        
        # Combine with the correct answer and shuffle
        options = selected_incorrect + [correct_answer]
        random.shuffle(options)
        return options
    
    def generate_image(self, prompt):
        """Generate an image using the Grok3 API"""
        if not self.api_key:
            return {
                "success": False,
                "error": "No API key provided"
            }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Note: As per memory, the Grok Image Generation API doesn't support the 'size' parameter
        payload = {
            "model": self.image_model,
            "prompt": prompt
        }
        
        try:
            print(f"Sending request to {GROK_API_URL} with payload: {payload}")
            response = requests.post(GROK_API_URL, headers=headers, json=payload)
            
            # Print response details for debugging
            print(f"Response status code: {response.status_code}")
            print(f"Response headers: {response.headers}")
            
            if response.status_code != 200:
                print(f"Error response content: {response.text}")
                return {
                    "success": False,
                    "error": f"{response.status_code} {response.reason}: {response.text}"
                }
            
            data = response.json()
            print(f"Response data keys: {data.keys()}")
            
            # Check if the response has the expected structure
            if "data" not in data or not data["data"] or "url" not in data["data"][0]:
                print(f"Unexpected response format: {data}")
                return {
                    "success": False,
                    "error": "Unexpected response format from API"
                }
            
            return {
                "success": True,
                "image_url": data["data"][0]["url"]
            }
        except requests.exceptions.RequestException as e:
            print(f"Request exception: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}"
            }
    
    def create_mystery(self, include_analysis=False):
        """Create a complete mystery challenge with optional AI analysis
        
        Args:
            include_analysis: If True, includes object detection and anachronism analysis
        """
        mystery = self.get_random_mystery()
        result = self.generate_image(mystery["prompt"])
        
        if not result["success"]:
            return result
        
        options = self.generate_options(mystery["anachronism"])
        
        response = {
            "success": True,
            "prompt": mystery["prompt"],
            "anachronism": mystery["anachronism"],
            "explanation": mystery["explanation"],
            "image_url": result["image_url"],
            "options": options
        }
        
        # Add AI analysis if requested
        if include_analysis:
            try:
                # Add object detection
                objects_result = self.detect_objects(result["image_url"])
                if objects_result["success"]:
                    response["object_detection"] = objects_result["objects"]
                
                # Add detailed anachronism analysis
                details_result = self.extract_anachronism_details(result["image_url"], mystery["anachronism"])
                if details_result["success"]:
                    response["anachronism_details"] = details_result["details"]
            except Exception as e:
                # Fail gracefully if analysis fails
                response["analysis_error"] = str(e)
        
        return response
    
    def check_answer(self, user_answer, correct_answer):
        """Check if the user's answer is correct"""
        return user_answer.lower() == correct_answer.lower()
    
    def detect_objects(self, image_url):
        """Use Grok-3 to detect objects in the generated image"""
        if not self.api_key:
            return {"success": False, "error": "No API key provided"}
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Download the image
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            image_data = response.content
            
            # Convert to base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Prepare the prompt for object detection
            messages = [
                {"role": "system", "content": "You are an expert at analyzing images and detecting objects. Identify all objects in the image and their approximate locations."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Analyze this image and list all objects you can identify. For each object, provide its name and approximate location (top-left, center, bottom-right, etc.)."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]}
            ]
            
            payload = {
                "model": self.chat_model,
                "messages": messages
            }
            
            response = requests.post(GROK_CHAT_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            return {
                "success": True,
                "objects": result["choices"][0]["message"]["content"]
            }
            
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}
    
    def extract_anachronism_details(self, image_url, anachronism):
        """Extract detailed information about the anachronism in the image"""
        if not self.api_key:
            return {"success": False, "error": "No API key provided"}
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        try:
            # Download the image
            response = requests.get(image_url)
            response.raise_for_status()
            image_data = response.content
            
            # Convert to base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # Prepare the prompt for structured data extraction
            messages = [
                {"role": "system", "content": "You are an expert at analyzing historical accuracy in images and identifying anachronisms."},
                {"role": "user", "content": [
                    {"type": "text", "text": f"In this image, there is an anachronism: {anachronism}. Please provide the following structured information:\n1. Detailed description of the {anachronism}\n2. Why it's out of place in this historical context\n3. When was this object/technology actually invented or became common\n4. What would be historically accurate instead"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]}
            ]
            
            payload = {
                "model": self.chat_model,
                "messages": messages
            }
            
            response = requests.post(GROK_CHAT_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            
            result = response.json()
            return {
                "success": True,
                "details": result["choices"][0]["message"]["content"]
            }
            
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": str(e)}

# Example usage
if __name__ == "__main__":
    # Create a .env file with your Grok3 API key or pass it directly
    bot = MysteryBot()
    
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