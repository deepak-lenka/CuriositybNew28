# Curiosity Blocks - Educational Topic Explorer

An interactive educational tool that generates and explores educational topics for students based on their grade level and curriculum board.

## Setup

1. Create a `.env` file in the root directory and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the App

Run the Streamlit app:
```bash
streamlit run app.py
```

## Features

- Generate educational topics based on grade level and board
- Explore detailed explanations of topics
- Discover related topics
- Maintain conversation context
- Reset conversation history

## Usage

1. Select your grade level and board from the sidebar
2. Click "Generate Topics" to get a list of educational topics
3. Enter a topic to explore and click "Explore Topic" to get detailed information
4. Click "Get Related Topics" to discover related educational topics
5. Use the "Reset Conversation" button to start fresh
