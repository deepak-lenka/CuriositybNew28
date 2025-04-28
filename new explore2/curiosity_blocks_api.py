import os
import json
import requests
import openai
import re
import threading
from typing import List, Dict, Any
from dotenv import load_dotenv
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# Load environment variables from .env file
load_dotenv()

# Set the API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = api_key

# Configure OpenAI client
client = openai.OpenAI(api_key=api_key)

class CuriosityBlocksAPI:
    def __init__(self):
        """
        Initialize the CuriosityBlocksAPI with LlamaIndex components and direct OpenAI API calls.
        """
        try:
            # Initialize the LLM for LlamaIndex
            self.llm = LlamaOpenAI(
                model="gpt-3.5-turbo-16k",
                temperature=0.7,
                max_tokens=4000,
            )
            
            # Initialize memory buffer
            self.memory = ChatMemoryBuffer.from_defaults(
                token_limit=3000,  # Limit token usage
            )
            
            # Create system prompt template
            self.system_prompt = (
                "You are an educational assistant for students. "
                "Be helpful, educational, and age-appropriate."
            )
            
            # Initialize document index if data directory exists
            try:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                data_dir = os.path.join(current_dir, "..", "data")
                
                # Create data directory if it doesn't exist
                if not os.path.exists(data_dir):
                    os.makedirs(data_dir, exist_ok=True)
                    print(f"Created data directory: {data_dir}")
                
                # Check if directory is empty
                if os.path.exists(data_dir) and not os.listdir(data_dir):
                    # Create a sample file with educational content
                    sample_file_path = os.path.join(data_dir, "sample_content.txt")
                    with open(sample_file_path, "w") as f:
                        f.write("""# Educational Content Sample
                        
This is a sample educational content file for the Curiosity Blocks application.

The application uses this file to initialize the document index when no other content is available.

Topics covered in this application include:
- Science (Physics, Chemistry, Biology, Astronomy)
- Mathematics (Algebra, Geometry, Calculus)
- History (Ancient, Medieval, Modern)
- Literature and Language Arts
- Geography and Social Studies
- Technology and Computer Science
                        """)
                    print(f"Created sample content file: {sample_file_path}")
                
                # Now load the data
                documents = SimpleDirectoryReader(data_dir).load_data()
                self.index = VectorStoreIndex.from_documents(documents)
                self.retriever = self.index.as_retriever()
            except Exception as e:
                print(f"Error initializing document index: {e}")
                self.retriever = None
            
            # Create chat engine with memory
            self.chat_engine = ContextChatEngine.from_defaults(
                llm=self.llm,
                memory=self.memory,
                system_prompt=self.system_prompt,
                retriever=self.retriever if hasattr(self, 'retriever') else None,
            )
            
            # Maintain conversation history manually for direct API calls
            self.conversation_history = [
                {"role": "system", "content": self.system_prompt}
            ]
            
            # Enhanced topic history and user interest tracking
            self.topic_history = []  # List of topics explored
            self.subtopic_history = []  # List of subtopics explored
            self.user_interests = {}  # Dictionary to track user interests and their weights
            self.interest_categories = {}  # Map topics to broader categories
            self.max_history = 10  # Keep more history for better personalization
            
            # Exa API configuration for web searches
            self.exa_api_key = os.getenv("EXA_API_KEY")
            self.exa_base_url = "https://api.exa.ai/search"
            print(f"Exa API Key loaded: {'*' if self.exa_api_key else 'NO API KEY'}")  # Debug print
        except Exception as e:
            print(f"Error initializing CuriosityBlocksAPI: {e}")
            raise

    def _get_completion_with_history(self, prompt: str, temperature: float = 0.7, max_tokens: int = 4000) -> str:
        """
        Get a completion using LlamaIndex chat engine with memory or fallback to direct OpenAI API.
        Includes topic history context to ensure continuity between topics.
        """
        # Add topic history context to the prompt
        topic_history_context = ""
        if self.topic_history:
            recent_topics = self.topic_history[-5:]  # Last 5 topics
            topic_history_context = f"\nPrevious topics explored: {', '.join(recent_topics)}.\n"
            topic_history_context += "Please ensure your response connects to these previous topics when relevant.\n"
        
        # Add educational framing for non-educational topics
        educational_framing = ""
        if hasattr(self, 'current_grade') and hasattr(self, 'current_board'):
            # Check if the prompt is about a non-educational topic
            educational_keywords = ['math', 'science', 'history', 'geography', 'literature', 'physics', 'chemistry', 'biology',
                                   'economics', 'politics', 'grammar', 'algebra', 'geometry', 'calculus', 'astronomy']
            contains_educational_keyword = any(keyword in prompt.lower() for keyword in educational_keywords)
            
            # If it's not clearly an educational topic, add educational framing
            if not contains_educational_keyword and hasattr(self, 'current_topic'):
                non_educational_keywords = ['movie', 'film', 'actor', 'actress', 'celebrity', 'entertainment', 'game', 'sport',
                                           'music', 'song', 'artist', 'band', 'tv', 'television', 'show', 'series']
                is_non_educational = any(keyword in self.current_topic.lower() for keyword in non_educational_keywords)
                
                if is_non_educational:
                    educational_framing = f"\nPresent this information in an educational way suitable for {self.current_grade} {self.current_board} students. "  
                    educational_framing += "Focus on learning opportunities, educational value, and connections to curriculum subjects. "
                    educational_framing += "Make sure the content is age-appropriate and has educational relevance.\n"
        
        enhanced_prompt = topic_history_context + educational_framing + prompt
        
        # Always use the direct OpenAI API approach since we're having issues with the retriever
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo-16k",
                messages=self.conversation_history + [{"role": "user", "content": enhanced_prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            reply = response.choices[0].message.content
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": enhanced_prompt})
            self.conversation_history.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            print(f"Error in OpenAI API call: {str(e)}")
            return "I'm having trouble processing that right now. Let's try something else."

    def generate_topics(self, grade: str, board: str, n_topics: int = 4) -> List[Dict[str, str]]:
        """
        Generate curiosity topics based on grade level, board, and user's exploration history.
        Ensures variety in suggested topics and avoids repetition.
        """
        try:
            # Get user interest context to personalize suggestions
            interest_context = self._get_user_interest_context() if self.user_interests else ""
            
            # Add excluded topics to avoid repetition
            excluded_topics = ""
            if self.topic_history:
                # Exclude recently explored topics to ensure fresh suggestions
                excluded_topics = f"\nDo NOT suggest any of these previously explored topics: {', '.join(self.topic_history[-10:])}\n"
            
            # Add randomization factor to ensure variety
            import time
            current_time = time.time()
            randomization = f"\nThe current timestamp is {current_time}. Use this as a seed to generate diverse topics.\n"
            
            # Build curriculum context based on grade and board
            curriculum_context = f"\nFor {grade} grade students in the {board} curriculum, focus on age-appropriate topics that align with their curriculum.\n"
            
            # Determine subject areas to focus on based on user history or provide variety
            subject_focus = ""
            if self.topic_history and len(self.topic_history) >= 2:
                # Infer learning focus from history
                learning_focus = self._infer_learning_focus(self.topic_history)
                subject_focus = f"\nBased on the user's exploration pattern, they seem to be {learning_focus}. "
                subject_focus += "Provide a mix of topics that both extend this focus and introduce complementary areas.\n"
            else:
                # If no history, ensure variety across subjects
                subject_focus = "\nProvide a diverse mix of topics across different subject areas (science, math, history, literature, arts, etc.).\n"
            
            prompt = f"""
            Generate {n_topics} interesting and DIVERSE educational topics for {grade} grade students.
            {curriculum_context}
            {interest_context}
            {subject_focus}
            {excluded_topics}
            {randomization}
            
            Each topic should be:
            1. Engaging and spark curiosity
            2. Educational and appropriate for {grade} grade
            3. Specific enough to explore in depth (avoid overly broad topics)
            4. DIFFERENT from each other (cover various subject areas)
            
            Format the response as a JSON array with {n_topics} topic objects, each containing:
            - topic (string): The name of the topic (max 50 chars)
            - description (string): A brief 1-sentence description (max 100 chars)
            """
            
            # Use higher temperature for more variety
            response = self._get_completion_with_history(prompt, temperature=0.9)
            try:
                topics = json.loads(response)
                # Validate the response format
                if not isinstance(topics, list):
                    raise ValueError("Response is not a list")
                
                formatted_topics = []
                for topic in topics:
                    if isinstance(topic, dict) and "topic" in topic and "description" in topic:
                        formatted_topics.append({
                            "topic": str(topic["topic"])[:50],  # Ensure string and length limit
                            "description": str(topic["description"])[:100]  # Ensure string and length limit
                        })
                
                return formatted_topics[:n_topics]  # Ensure we return exactly n_topics
            except json.JSONDecodeError as e:
                print(f"Error parsing topics JSON: {e}")
                print(f"Raw response: {response}")
                # Return a fallback response
                return [{"topic": f"Topic {i+1}", "description": "Generic topic description"} for i in range(n_topics)]
        except Exception as e:
            print(f"Error in generate_topics: {e}")
            raise

    def explain_topic(self, topic: str, grade: str, board: str) -> Dict[str, Any]:
        """
        Generate a detailed explanation for a given topic.
        
        Args:
            topic: The topic to explain
            grade: The grade level of the user
            board: The educational board/curriculum
            
        Returns:
            A dictionary containing the main topic content, subtopics, and related topics
        """
        # Store current topic, grade and board for content filtering and fallback searches
        self.current_topic = topic
        self.current_grade = grade
        self.current_board = board
        
        # Store the topic category for better image selection
        topic_categories = {
            'science': ['physics', 'chemistry', 'biology', 'astronomy', 'space', 'earth', 'environment', 'technology', 'engineering'],
            'math': ['mathematics', 'algebra', 'geometry', 'calculus', 'statistics', 'arithmetic', 'number', 'equation'],
            'history': ['history', 'civilization', 'war', 'ancient', 'medieval', 'modern', 'revolution', 'empire', 'kingdom'],
            'geography': ['geography', 'map', 'country', 'continent', 'ocean', 'river', 'mountain', 'climate', 'weather'],
            'literature': ['literature', 'book', 'novel', 'poem', 'author', 'writer', 'story', 'character', 'fiction'],
            'art': ['art', 'painting', 'sculpture', 'music', 'dance', 'theater', 'film', 'photography', 'design'],
        }
        
        # Determine the category of the topic
        topic_lower = topic.lower()
        self.topic_category = 'general'
        for category, keywords in topic_categories.items():
            if any(keyword in topic_lower for keyword in keywords):
                self.topic_category = category
                break
        
        # Update topic history and user interests
        try:
            self._manage_topic_history(topic)
            interest_context = self._get_user_interest_context()
        except Exception as e:
            print(f"Error preparing topic context: {e}")
            interest_context = ""
        
        # Start web search in parallel with explanation generation
        import threading
        self.main_topic_results = []
        
        def background_search():
            # Determine if the topic is educational or general interest
            educational_keywords = ['math', 'science', 'history', 'geography', 'literature', 'physics', 'chemistry', 'biology', 
                                  'economics', 'politics', 'grammar', 'algebra', 'geometry', 'calculus', 'astronomy']
            
            is_educational_topic = any(keyword in topic.lower() for keyword in educational_keywords)
            
            if is_educational_topic:
                # For educational topics, use grade and board in the query
                main_topic_query = f"{topic} for {grade} {board} educational resources classroom teaching images"
                self.main_topic_results = self._search_web(main_topic_query, educational_focus=True)
            else:
                # For non-educational topics (movies, etc.), first get general information
                general_query = f"{topic} information facts summary"
                general_results = self._search_web(general_query, educational_focus=False)
                
                # Then get educational perspective on the same topic
                educational_query = f"{topic} educational perspective for {grade} students learning material"
                educational_results = self._search_web(educational_query, educational_focus=True)
                
                # Combine results, with educational ones first
                self.main_topic_results = educational_results + [r for r in general_results if r not in educational_results]
                
                # Limit to top results
                self.main_topic_results = self.main_topic_results[:5]
            
        # Start the search in a background thread
        search_thread = threading.Thread(target=background_search)
        search_thread.daemon = True
        search_thread.start()

        # Generate the explanation
        try:
            # Prepare prompt for generating content with personalization
            # Analyze why the user might be choosing this topic based on previous exploration
            previous_topics_context = ""
            user_intent_analysis = ""
            
            if self.topic_history and len(self.topic_history) > 1:
                # Get previous topics excluding the current one
                prev_topics = [t for t in self.topic_history if t.lower() != topic.lower()]
                if prev_topics:
                    most_recent = prev_topics[-1] if prev_topics else ""
                    
                    # Analyze potential user intent based on topic progression
                    user_intent_analysis = f"\nThe user has moved from exploring '{most_recent}' to '{topic}'. "
                    user_intent_analysis += f"This suggests they may be:"
                    user_intent_analysis += f"\n1. Building on knowledge from '{most_recent}' to understand '{topic}'"
                    user_intent_analysis += f"\n2. Comparing or contrasting '{most_recent}' with '{topic}'"
                    user_intent_analysis += f"\n3. Exploring a specific aspect mentioned in '{most_recent}' in more depth"
                    user_intent_analysis += f"\n4. Following a logical learning progression in this subject area\n"
                    
                    # Create explicit connection instructions
                    previous_topics_context = f"\nThe user recently explored '{most_recent}'. "
                    previous_topics_context += f"Please make explicit connections between '{topic}' and '{most_recent}' in your explanation, "
                    previous_topics_context += f"addressing why the user might have chosen to explore '{topic}' after '{most_recent}'.\n"
                    
                    if len(prev_topics) > 1:
                        other_topics = prev_topics[:-1]
                        previous_topics_context += f"\nThe user's broader exploration journey includes: {', '.join(other_topics)}. "
                        previous_topics_context += f"This suggests a learning path focused on {self._infer_learning_focus(prev_topics + [topic])}. "
                        previous_topics_context += "Connect your explanation to this broader learning journey.\n"
            
            prompt = f"""
            Create an educational explanation about "{topic}" for {grade} grade students following the {board} curriculum.
            The explanation should be engaging, informative, and appropriate for their grade level.
            
            {interest_context}
            {previous_topics_context}
            {user_intent_analysis}
            
            Please tailor the explanation to connect with the user's interests and previous topics where relevant, 
            while maintaining educational accuracy. Make sure to draw explicit connections between this topic 
            and previously explored topics. Address why the user might have chosen to explore this topic based on their learning journey.
            
            Structure the response as a JSON object with the following sections:
            1. main_topic: An object containing:
               - title: A catchy title for the topic
               - explanation: A detailed, engaging explanation (300-500 words) that connects to the user's interests and previous topics where relevant
               - image_url: (optional) A URL to a relevant educational image
            
            2. subtopics: An array of 2-3 objects, each containing:
               - title: A clear title for the subtopic
               - explanation: A concise explanation (100-150 words)
               - web_resources: (optional) Links to educational resources
            
            3. related_topics: An array of 2-3 objects, each containing:
               - topic: Name of a related topic that connects to both the main topic and the user's previous explorations
               - summary: A brief summary of how it relates (1-2 sentences)
               - web_resources: (optional) Links to educational resources
            """
            
            # Get response from OpenAI
            response = self._get_completion_with_history(prompt)
            
            # Parse the JSON response
            try:
                content = json.loads(response)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response: {e}")
                print(f"Raw response: {response}")
                return {
                    "main_topic": {"title": topic, "explanation": "Failed to parse the explanation. Please try again."}, 
                    "subtopics": [], 
                    "related_topics": []
                }
            
            # Process main topic explanation
            main_topic = content["main_topic"]
            explanation = main_topic["explanation"]
            
            # Ensure explanation is at least 100 words and ends with a complete sentence
            words = explanation.split()
            if len(words) < 100:
                # If less than 100 words, keep adding words until we reach at least 100
                # or until we reach the end of a complete sentence
                sentences = explanation.split('.')
                new_explanation = []
                word_count = 0
                
                for sentence in sentences:
                    sentence_words = sentence.split()
                    word_count += len(sentence_words)
                    new_explanation.append(sentence)
                    if word_count >= 100:
                        break
                
                # Join sentences and add period
                explanation = '.'.join(new_explanation).strip() + '.'
            else:
                # If more than 100 words, find the last complete sentence before 100 words
                current_length = 0
                sentences = explanation.split('.')
                new_explanation = []
                
                for sentence in sentences:
                    sentence_words = sentence.split()
                    if current_length + len(sentence_words) > 100:
                        break
                    current_length += len(sentence_words)
                    new_explanation.append(sentence)
                
                # Join sentences and add period
                explanation = '.'.join(new_explanation).strip() + '.'

            main_topic["explanation"] = explanation
            
            # Ensure we have exactly 3 related topics
            related_topics = content["related_topics"]
            if len(related_topics) > 3:
                related_topics = related_topics[:3]
            elif len(related_topics) < 3:
                # Use topic history to fill in missing related topics instead of placeholders
                # Get previous topics from history (excluding current topic)
                previous_topics = [t for t in self.topic_history if t.lower() != topic.lower()]
                
                # If we have previous topics, use them
                if previous_topics:
                    for i in range(3 - len(related_topics)):
                        if i < len(previous_topics):
                            # Use a previous topic with a relevant connection
                            prev_topic = previous_topics[i]
                            related_topics.append({
                                "topic": prev_topic,
                                "summary": f"This topic is related to {topic} and was previously explored. Revisiting it may help deepen your understanding of both subjects."
                            })
                        else:
                            # If we run out of previous topics, suggest a related field
                            related_topics.append({
                                "topic": self._generate_related_field(topic),
                                "summary": f"This field is connected to {topic} and exploring it will broaden your understanding of the subject area."
                            })
                else:
                    # If no history, generate related fields based on the current topic
                    for i in range(3 - len(related_topics)):
                        related_topics.append({
                            "topic": self._generate_related_field(topic),
                            "summary": f"This field is connected to {topic} and exploring it will broaden your understanding of the subject area."
                        })
            
            # Format web results to be concise
            def format_web_result(result):
                summary = result.get('summary', '')
                highlights = result.get('highlights', [])
                
                # Truncate summary to 40-50 words
                summary_words = summary.split()
                if len(summary_words) > 50:
                    summary = ' '.join(summary_words[:50]) + "..."
                elif len(summary_words) < 40:
                    summary = ' '.join(summary_words) + "..."
                
                # Truncate highlights to 40-50 words
                formatted_highlights = []
                for highlight in highlights[:2]:
                    highlight_words = highlight.split()
                    if len(highlight_words) > 50:
                        highlight = ' '.join(highlight_words[:50]) + "..."
                    elif len(highlight_words) < 40:
                        highlight = ' '.join(highlight_words) + "..."
                    formatted_highlights.append(highlight)
                
                result['summary'] = summary
                result['highlights'] = formatted_highlights
                return result

            # Use the results from the parallel search that started earlier
            # Wait for a maximum of 3 seconds for the thread to complete
            import time
            start_time = time.time()
            while not self.main_topic_results and time.time() - start_time < 3:
                time.sleep(0.1)
                
            # Use whatever results we have, even if the search is still running
            web_resources = self._format_web_results(self.main_topic_results)
            main_topic['web_resources'] = web_resources['formatted_text']
            main_topic['image_url'] = web_resources['image_url']
            
            # Get web resources for subtopics
            for subtopic in content["subtopics"]:
                try:
                    # Determine if the subtopic is educational or general interest
                    educational_keywords = ['math', 'science', 'history', 'geography', 'literature', 'physics', 'chemistry', 'biology', 
                                          'economics', 'politics', 'grammar', 'algebra', 'geometry', 'calculus', 'astronomy']
                    
                    is_educational_subtopic = any(keyword in subtopic['title'].lower() for keyword in educational_keywords)
                    
                    if is_educational_subtopic:
                        # For educational subtopics, use grade and board in the query
                        subtopic_query = f"{subtopic['title']} for {grade} {board} classroom teaching images educational resources"
                        subtopic_results = self._search_web(subtopic_query, educational_focus=True)
                    else:
                        # For non-educational subtopics, get educational perspective
                        subtopic_query = f"{subtopic['title']} educational perspective for {grade} students learning material"
                        subtopic_results = self._search_web(subtopic_query, educational_focus=True)
                    subtopic['web_resources'] = self._format_web_results(subtopic_results)['formatted_text']
                except Exception as e:
                    print(f"Error getting web resources for subtopic: {e}")
                    subtopic['web_resources'] = ""

            return {
                "main_topic": main_topic,
                "subtopics": content["subtopics"],
                "related_topics": related_topics
            }
            
        except Exception as e:
            print(f"Error in explain_topic: {e}")
            return {
                "main_topic": {"title": topic, "explanation": "Failed to generate explanation due to an error."}, 
                "subtopics": [], 
                "related_topics": []
            }

    def explore_related_topics(self, topic: str, grade: str, board: str) -> List[Dict[str, str]]:
        """
        Generate personalized related topics based on the current topic and user interests.
        """
        try:
            # Update topic history and user interests
            self._manage_topic_history(topic)
            
            # Get user interest context
            interest_context = self._get_user_interest_context()
            
            # Get ALL previous topics to explicitly connect with
            previous_topics_context = ""
            if self.topic_history and len(self.topic_history) > 1:
                # Get previous topics excluding the current one
                prev_topics = [t for t in self.topic_history if t.lower() != topic.lower()]
                if prev_topics:
                    all_prev_topics = ", ".join(prev_topics)
                    
                    previous_topics_context = f"\nCRITICAL INSTRUCTION: The user has previously explored these topics: {all_prev_topics}. "
                    previous_topics_context += f"Now they are exploring '{topic}'.\n\n"
                    previous_topics_context += f"Your task is to create CREATIVE and EDUCATIONAL connections between '{topic}' and ALL previously explored topics.\n"
                    previous_topics_context += f"Even if the topics seem completely unrelated (like 'Moon' and 'Cybersecurity'), find innovative ways to connect them.\n"
                    previous_topics_context += f"For example, if previous topics were 'Moon' and 'Cybersecurity', you might suggest 'Satellite Security: Protecting Lunar Communication Systems'.\n"
            
            # Build a more personalized prompt based on user interests and previous topics
            prompt = f"""
            Based on the topic "{topic}" and ALL previously explored topics, suggest exactly 3 creative and educational topics for {grade} grade students.
            
            {interest_context}
            {previous_topics_context}
            
            CRITICAL REQUIREMENTS:
            1. Each topic MUST create connections between the current topic and AT LEAST ONE previously explored topic
            2. Be CREATIVE and INNOVATIVE in finding connections between seemingly unrelated topics
            3. The connections should be educational, meaningful, and provide new insights
            4. Topics should be appropriate for {grade} grade students and expand their understanding
            5. If possible, try to connect multiple previous topics together with the current topic
            
            For each topic, provide:
            1. A clear and engaging title that shows the creative connection (max 50 chars)
            2. A brief description of how this topic connects previously explored topics (max 100 chars)
            
            RESPOND IN THIS EXACT FORMAT - A JSON ARRAY WITH 3 OBJECTS:
            [
              {{
                "topic": "Topic Title",
                "description": "Brief description of connection"
              }},
              {{
                "topic": "Topic Title 2",
                "description": "Brief description of connection 2"
              }},
              {{
                "topic": "Topic Title 3",
                "description": "Brief description of connection 3"
              }}
            ]
            
            ONLY RETURN THE JSON. NO OTHER TEXT.
            
            Format as a JSON array with exactly 3 objects, each containing:
            - topic (string): Related topic name (max 50 chars)
            - description (string): A concise explanation of why this topic is relevant
            - connection (string): How this connects to the user's previous explorations
            """
            # Use GPT-4.1 specifically for related topics to get faster responses
            try:
                response = client.chat.completions.create(
                    model="gpt-4-1106-preview",  # Use GPT-4.1 for faster, more creative responses
                    messages=[{"role": "system", "content": self.system_prompt}, {"role": "user", "content": prompt}],
                    temperature=0.9,  # Higher temperature for more creative connections
                    max_tokens=1000,  # Shorter response for faster generation
                )
                response = response.choices[0].message.content
                print(f"Using GPT-4.1 for related topics generation")
            except Exception as e:
                print(f"Error using GPT-4.1, falling back to default model: {e}")
                response = self._get_completion_with_history(prompt)
            try:
                # Try to extract JSON from the response if it's not pure JSON
                if not response.strip().startswith('['):
                    # Look for JSON array in the response
                    json_match = re.search(r'\[\s*\{.*?\}\s*\]', response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        print(f"Extracted JSON from response: {json_str}")
                        related_topics = json.loads(json_str)
                    else:
                        # Try to find JSON in code blocks
                        json_match = re.search(r'```(?:json)?\s*\n?(\[\s*\{.*?\}\s*\])\s*```', response, re.DOTALL)
                        if json_match:
                            json_str = json_match.group(1)
                            print(f"Extracted JSON from code block: {json_str}")
                            related_topics = json.loads(json_str)
                        else:
                            # If no JSON found, manually parse the response
                            print("No JSON found in response, manually parsing...")
                            print(f"Raw response: {response}")
                            # Create a simple structure with topic titles and descriptions
                            related_topics = self._manually_parse_topics(response)
                else:
                    # It's already in JSON format
                    related_topics = json.loads(response)
                
                # Validate and format the response
                formatted_topics = []
                for rt in related_topics:
                    if isinstance(rt, dict):
                        topic = str(rt.get("topic", ""))[:50]  # Ensure string and length limit
                        
                        # Check if the API returned description or summary (handle both formats)
                        description = rt.get("description", rt.get("summary", "No description provided."))
                        description = str(description)
                        
                        # Get connection to user interests if available
                        connection = rt.get("connection", "")
                        if connection:
                            # Don't just append the connection - make it the focus
                            if "connection" in rt and len(connection) > len(description):
                                description = connection
                            else:
                                description += f" {connection}"
                            
                        formatted_topics.append({
                            "topic": topic if topic else f"Related Topic {len(formatted_topics) + 1}",
                            "description": description
                        })
                
                # Ensure exactly 3 topics by using previously explored topics or related fields
                if len(formatted_topics) < 3:
                    # Get previous topics from history (excluding current topic)
                    previous_topics = [t for t in self.topic_history if t.lower() != topic.lower()]
                    
                    # If we have previous topics, use them
                    if previous_topics:
                        for i in range(3 - len(formatted_topics)):
                            if i < len(previous_topics):
                                # Use a previous topic with a relevant connection
                                prev_topic = previous_topics[i]
                                formatted_topics.append({
                                    "topic": prev_topic,
                                    "description": f"You previously explored this topic. Revisiting {prev_topic} may help deepen your understanding and see connections with {topic}."
                                })
                            else:
                                # If we run out of previous topics, suggest a related field
                                related_field = self._generate_related_field(topic)
                                formatted_topics.append({
                                    "topic": related_field,
                                    "description": f"This field is connected to {topic} and exploring it will broaden your understanding of the subject area."
                                })
                    else:
                        # If no history, generate related fields based on the current topic
                        for i in range(3 - len(formatted_topics)):
                            related_field = self._generate_related_field(topic)
                            formatted_topics.append({
                                "topic": related_field,
                                "description": f"This field is connected to {topic} and exploring it will broaden your understanding of the subject area."
                            })
                
                return formatted_topics[:3]  # Return exactly 3 topics
            except json.JSONDecodeError as e:
                print(f"Error parsing related topics JSON: {e}")
                print(f"Raw response: {response}")
                # Try manual parsing as a fallback
                related_topics = self._manually_parse_topics(response)
                if not related_topics:
                    return []  # Return an empty list instead of a fallback response
                return [
                    {"topic": f"Related Topic {i+1}", "description": "Related topic description"}
                    for i in range(3)
                ]
        except Exception as e:
            print(f"Error in explore_related_topics: {e}")
            raise

    def _manually_parse_topics(self, text: str) -> List[Dict[str, str]]:
        """Manually parse topics from text when JSON parsing fails."""
        import re
        topics = []
        
        # Look for numbered topics (1. Topic: "Title" or 1. "Title" or Topic 1: "Title")
        topic_patterns = [
            r'\d+\.\s*(?:Topic:|")(.*?)(?:"|-)\s*(?:-|:)\s*(.*?)(?=\d+\.|$|\n\n)',  # 1. Topic: "Title" - Description
            r'\d+\.\s*"(.*?)"\s*(?:-|:)\s*(.*?)(?=\d+\.|$|\n\n)',  # 1. "Title" - Description
            r'Topic\s*\d+:?\s*"?(.*?)"?\s*(?:-|:)\s*(.*?)(?=Topic\s*\d+|$|\n\n)'  # Topic 1: Title - Description
        ]
        
        for pattern in topic_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                for match in matches[:3]:  # Take up to 3 topics
                    topic = match[0].strip()
                    description = match[1].strip() if len(match) > 1 else ""
                    topics.append({"topic": topic, "description": description})
                break
        
        # If no matches found, try to extract any topic-like sections
        if not topics:
            # Look for sections that might contain topics
            sections = re.split(r'\n\s*\n', text)
            for i, section in enumerate(sections[:3]):  # Take up to 3 sections
                # Try to extract a title and description
                title_match = re.search(r'(?:"|")([^"]+)(?:"|"|:)|([A-Z][^\n:]+):', section)
                if title_match:
                    title = title_match.group(1) or title_match.group(2)
                    # Get the rest as description
                    desc_text = re.sub(r'(?:"|")([^"]+)(?:"|"|:)|([A-Z][^\n:]+):', '', section, 1)
                    topics.append({"topic": title.strip(), "description": desc_text.strip()})
                else:
                    # Just use the first line as title and the rest as description
                    lines = section.strip().split('\n', 1)
                    title = lines[0].strip()
                    desc = lines[1].strip() if len(lines) > 1 else ""
                    topics.append({"topic": title, "description": desc})
        
        # If we still don't have topics, create generic ones
        if not topics:
            for i in range(1, 4):  # Create 3 generic topics
                topics.append({
                    "topic": f"Related Topic {i}",
                    "description": f"A topic related to your exploration journey."
                })
        
        return topics

    def _generate_related_field(self, topic: str) -> str:
        """Generate a related field to a topic when we need to fill in additional topics."""
        fields = {
            "math": ["algebra", "geometry", "statistics", "calculus"],
            "science": ["physics", "chemistry", "biology", "astronomy"],
            "history": ["ancient history", "world history", "modern history", "cultural history"],
            "literature": ["poetry", "fiction", "drama", "literary analysis"],
            "art": ["painting", "sculpture", "photography", "digital art"],
            "music": ["classical music", "music theory", "music history", "music composition"],
            "technology": ["computer science", "programming", "artificial intelligence", "robotics"],
            "geography": ["physical geography", "human geography", "cartography", "geology"]
        }
        
        # Try to match the topic to a field
        topic_lower = topic.lower()
        matched_fields = []
        
        for field, subjects in fields.items():
            if field in topic_lower or any(subject in topic_lower for subject in subjects):
                # Don't suggest subjects from the same field
                other_fields = [f for f in fields.keys() if f != field]
                if other_fields:
                    # Get a random field that's different from the matched one
                    import random
                    random_field = random.choice(other_fields)
                    random_subject = random.choice(fields[random_field])
                    matched_fields.append(random_subject)
        
        if matched_fields:
            return matched_fields[0]
        else:
            # If no match found, return a random educational field
            import random
            random_field = random.choice(list(fields.keys()))
            return random.choice(fields[random_field])

    def _search_web(self, query: str, content_filter: str = "medium", educational_focus: bool = False) -> List[Dict[str, Any]]:
        """
        Search the web using Exa AI's search API with content filtering based on grade level.
        
        Args:
            query: The search query string
            content_filter: Level of content filtering ("none", "low", "medium", "high")
                - "low": Allow most content
                - "medium": Filter out adult content (default)
                - "high": Filter out adult content and potentially sensitive topics
            educational_focus: Whether to prioritize educational domains in search results
        """
        if not self.exa_api_key:
            print("No EXA_API_KEY found in environment variables")
            return []

        headers = {
            "Authorization": f"Bearer {self.exa_api_key}",
            "Content-Type": "application/json"
        }
        
        # Determine content filter level based on grade
        # Extract grade number if it's in format like "Grade 10" or "10"  
        grade_num = 0
        if hasattr(self, 'current_grade') and self.current_grade:
            grade_str = self.current_grade.lower().replace('grade', '').strip()
            try:
                grade_num = int(grade_str)
            except (ValueError, TypeError):
                grade_num = 0
        
        # Set content filter based on grade level
        if grade_num < 6:  # Elementary school
            content_filter = "high"
        elif grade_num < 9:  # Middle school
            content_filter = "medium"
        else:  # High school and above
            content_filter = "medium"
            
        # If educational focus is requested, modify the query to prioritize educational content
        if educational_focus:
            # Add educational domains to search
            educational_domains = ["edu", "org", "gov", "k12", "ac.in", "nic.in", "cbse", "icse", "ncert"]
            
            # If board is specified, add it to the domains (e.g., CBSE, ICSE)
            if hasattr(self, 'current_board') and self.current_board:
                board_domain = self.current_board.lower().replace(' ', '')
                if board_domain and board_domain not in educational_domains:
                    educational_domains.append(board_domain)
            
        payload = {
            "query": query,
            "type": "neural",  # Use neural search for faster results
            "numResults": 10,  # Increase results to get more image options
            "safeSearch": content_filter,  # Apply content filtering based on grade level
            "contents": {
                "text": True,
                "highlights": True,
                "summary": True,
                "image": True,  # Keep image content
                "livecrawl": "fallback",  # Only use livecrawl as fallback
                "livecrawlTimeout": 3000  # Reduce timeout to 3 seconds
            }
        }
        
        # If educational focus is requested, add domain filtering
        if educational_focus and hasattr(self, 'current_grade') and hasattr(self, 'current_board'):
            # Add educational domains to search
            payload["useAuthorityFilter"] = True
            
            # Check if the query is about a non-educational topic
            educational_keywords = ['math', 'science', 'history', 'geography', 'literature', 'physics', 'chemistry', 'biology',
                                  'economics', 'politics', 'grammar', 'algebra', 'geometry', 'calculus', 'astronomy']
            is_educational_query = any(keyword in query.lower() for keyword in educational_keywords)
            
            # For non-educational topics, add educational framing
            if not is_educational_query:
                payload["query"] = f"{payload['query']} educational perspective learning material"
            
            # Add specific query modifiers for educational content based on board
            if 'ICSE' in query or (hasattr(self, 'current_board') and 'ICSE' in self.current_board):
                payload["query"] = f"{payload['query']} ICSE curriculum teaching materials"
            elif 'CBSE' in query or (hasattr(self, 'current_board') and 'CBSE' in self.current_board):
                payload["query"] = f"{payload['query']} CBSE curriculum teaching materials"

        try:
            print(f"Making Exa API request for query: {query}")
            response = requests.post(
                self.exa_base_url,
                headers=headers,
                json=payload,
                timeout=10  # Use a 10-second timeout for reliability
            )
            
            print(f"Exa API response status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                print(f"Exa API response: {json.dumps(data, indent=2)}")
                
                # Filter out low-quality results
                results = data.get("results", [])
                filtered_results = []
                for result in results:
                    if result.get("score", 0) > 0.3 and result.get("text") and result.get("url"):
                        # Add debug info about image presence
                        has_image = bool(result.get('image') or result.get('thumbnail') or result.get('imageUrl') or result.get('favicon'))
                        print(f"Result has image: {has_image}")
                        
                        # Calculate relevance score for better image selection
                        relevance_score = result.get("score", 0)
                        
                        # Boost score for educational domains
                        url = result.get('url', '')
                        if any(domain in url for domain in ['edu', 'org', 'gov', 'school', 'learn', 'teach', 'cbse', 'icse', 'ncert']):
                            relevance_score += 0.2
                        
                        # Boost score for results with good images
                        if result.get('image') and isinstance(result.get('image'), str):
                            # Check if image URL contains educational keywords
                            image_url = result.get('image')
                            print(f"Found image URL: {image_url}")
                            if any(keyword in image_url.lower() for keyword in ['school', 'class', 'education', 'learn', 'teach', 'student']):
                                relevance_score += 0.1
                        
                        # Store the calculated relevance score
                        result['relevance_score'] = relevance_score
                        filtered_results.append(result)
                
                # Sort results by relevance score
                filtered_results.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
                
                # Return up to 5 filtered results
                return filtered_results[:5]
            else:
                print(f"Exa API error: {response.text}")
                return []
            
        except requests.exceptions.Timeout:
            print("Exa API request timed out")
            return []
        except requests.exceptions.RequestException as e:
            print(f"Error making Exa API request: {e}")
            return []

    def _format_web_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format web search results into a readable string with categorized links and extract image URL.
        """
        if not results:
            # Even if no results, provide a default image based on the current topic
            default_image = None
            if hasattr(self, 'current_topic'):
                # Default educational images by category
                default_images = {
                    'science': 'https://cdn.pixabay.com/photo/2018/09/27/09/22/artificial-intelligence-3706562_1280.jpg',
                    'math': 'https://cdn.pixabay.com/photo/2015/11/15/07/47/geometry-1044090_1280.jpg',
                    'history': 'https://cdn.pixabay.com/photo/2018/08/15/07/19/indian-history-3607311_1280.jpg',
                    'geography': 'https://cdn.pixabay.com/photo/2016/10/30/20/22/map-1784029_1280.jpg',
                    'literature': 'https://cdn.pixabay.com/photo/2015/11/19/21/10/glasses-1052010_1280.jpg',
                    'art': 'https://cdn.pixabay.com/photo/2016/06/25/12/55/art-1478831_1280.jpg',
                    'default': 'https://cdn.pixabay.com/photo/2015/07/31/11/45/library-869061_1280.jpg'  # General education image
                }
                
                # Simple categorization
                topic_lower = self.current_topic.lower()
                if any(term in topic_lower for term in ['physics', 'chemistry', 'biology', 'science', 'technology']):
                    default_image = default_images['science']
                elif any(term in topic_lower for term in ['math', 'algebra', 'geometry', 'calculus']):
                    default_image = default_images['math']
                elif any(term in topic_lower for term in ['history', 'war', 'civilization', 'empire']):
                    default_image = default_images['history']
                elif any(term in topic_lower for term in ['geography', 'map', 'country', 'world']):
                    default_image = default_images['geography']
                elif any(term in topic_lower for term in ['literature', 'book', 'novel', 'poem']):
                    default_image = default_images['literature']
                elif any(term in topic_lower for term in ['art', 'music', 'painting', 'dance']):
                    default_image = default_images['art']
                else:
                    default_image = default_images['default']
            
            return {
                "formatted_text": "No additional web resources found.",
                "image_url": default_image
            }

        # Categorize results by domain type
        categories = {
            "YouTube": [],
            "Twitter/X": [],
            "Wikipedia": [],
            "News": [],
            "Educational": [],
            "Other": []
        }

        # Show exactly 5 results total
        total_count = 0
        for result in results[:10]:  # Look at top 10 results
            if total_count >= 5:
                break
            
            url = result.get('url', '')
            domain = url.split('/')[2] if '/' in url else url
            
            # Categorize based on domain
            if 'youtube.com' in domain or 'youtu.be' in domain:
                if len(categories["YouTube"]) < 1:  # Limit to 1 YouTube video
                    categories["YouTube"].append(result)
                    total_count += 1
            elif 'twitter.com' in domain or 'x.com' in domain:
                if len(categories["Twitter/X"]) < 1:  # Limit to 1 Twitter/X post
                    categories["Twitter/X"].append(result)
                    total_count += 1
            elif 'wikipedia.org' in domain:
                if len(categories["Wikipedia"]) < 1:  # Limit to 1 Wikipedia article
                    categories["Wikipedia"].append(result)
                    total_count += 1
            elif any(site in domain for site in ['news', 'times', 'post', 'journal', 'reuters', 'bbc', 'cnn']):
                if len(categories["News"]) < 1:  # Limit to 1 news article
                    categories["News"].append(result)
                    total_count += 1
            elif any(site in domain for site in ['edu', 'org', 'gov']):
                if len(categories["Educational"]) < 1:  # Limit to 1 educational resource
                    categories["Educational"].append(result)
                    total_count += 1
            else:
                if len(categories["Other"]) < 1:  # Limit to 1 other resource
                    categories["Other"].append(result)
                    total_count += 1

        # Try to find an image URL from the results that's appropriate for the grade level
        image_url = None
        
        # Define educational and classroom-related keywords to prioritize images
        educational_keywords = ['classroom', 'school', 'education', 'learning', 'teaching', 'student', 
                               'lesson', 'textbook', 'curriculum', 'academic', 'study', 'educational',
                               'board', 'cbse', 'icse', 'ncert', 'diagram', 'illustration']
        
        # Get the current grade and board for more specific matching
        current_grade = getattr(self, 'current_grade', '').lower()
        current_board = getattr(self, 'current_board', '').lower()
        
        # First, analyze and score all results with images
        scored_results = []
        for r in results:
            # Only consider results with images
            has_image = bool(r.get('image') or r.get('thumbnail') or r.get('imageUrl') or r.get('favicon'))
            if not has_image:
                continue
                
            url = r.get('url', '').lower()
            title = r.get('title', '').lower()
            text = r.get('text', '').lower()
            summary = r.get('summary', '').lower()
            
            # Initialize comprehensive scoring system
            image_score = r.get('relevance_score', 0) * 10  # Base score from API relevance
            
            # 1. Domain authority scoring
            domain_score = 0
            if any(domain in url for domain in ['edu', 'ac.in', 'nic.in', 'gov']):
                domain_score = 3  # Highest priority for educational institutions
            elif any(domain in url for domain in ['org', 'school', 'learn', 'teach']):
                domain_score = 2  # High priority for educational organizations
            elif any(domain in url for domain in ['com/education', 'wikipedia']):
                domain_score = 1  # Medium priority for commercial educational sites
            image_score += domain_score * 5
            
            # 2. Content relevance scoring
            content_score = 0
            # Check for educational keywords in title (weighted highest)
            if any(keyword in title for keyword in educational_keywords):
                content_score += 3
            # Check for educational keywords in summary (weighted medium)
            if any(keyword in summary for keyword in educational_keywords):
                content_score += 2
            # Check for educational keywords in text (weighted lower)
            if any(keyword in text[:500] for keyword in educational_keywords):
                content_score += 1
            image_score += content_score * 3
            
            # 3. Grade and board specific scoring
            specificity_score = 0
            # Check for grade match
            if current_grade and (current_grade in title or current_grade in summary or current_grade in text[:500]):
                specificity_score += 3
            # Check for board match (CBSE, ICSE, etc.)
            if current_board and (current_board in title or current_board in summary or current_board in text[:500]):
                specificity_score += 4
            image_score += specificity_score * 4
            
            # 4. Image quality heuristics
            image_quality_score = 0
            # Prefer image URLs that contain educational terms
            image_url_to_check = r.get('image') or r.get('thumbnail') or r.get('imageUrl') or ''
            if isinstance(image_url_to_check, str):
                if any(term in image_url_to_check.lower() for term in ['diagram', 'illustration', 'figure', 'chart']):
                    image_quality_score += 2
                if any(term in image_url_to_check.lower() for term in educational_keywords):
                    image_quality_score += 1
            image_score += image_quality_score * 2
            
            # Store the final score
            r['image_score'] = image_score
            scored_results.append(r)
        
        # Sort by comprehensive image score (higher is better)
        scored_results.sort(key=lambda x: x.get('image_score', 0), reverse=True)
        
        # Log the top 3 results for debugging
        for i, result in enumerate(scored_results[:3]):
            print(f"Top image candidate {i+1}: Score={result.get('image_score', 0)}, URL={result.get('url', '')}, Title={result.get('title', '')}")
        
        # Use the sorted results as our prioritized list
        prioritized_results = scored_results
        
        # First pass: Try to find high-quality images (non-favicon)
        for result in prioritized_results:
            # Check if the result has an image URL in various fields and formats
            if isinstance(result.get('image'), str) and result.get('image').startswith(('http://', 'https://')):
                # Verify the image URL doesn't contain terms that suggest it's an icon or logo
                image_lower = result.get('image').lower()
                if not any(term in image_lower for term in ['icon', 'logo', 'favicon', 'avatar', 'button']):
                    image_url = result['image']
                    print(f"Selected image URL: {image_url} (score: {result.get('image_score', 0)})")
                    # Store additional metadata about the selected image for debugging
                    print(f"Image source: {result.get('url', 'unknown')}")
                    print(f"Image title: {result.get('title', 'unknown')}")
                    break
            elif result.get('thumbnail') and isinstance(result.get('thumbnail'), str) and result.get('thumbnail').startswith(('http://', 'https://')):
                # Verify the thumbnail URL doesn't contain terms that suggest it's an icon or logo
                thumb_lower = result.get('thumbnail').lower()
                if not any(term in thumb_lower for term in ['icon', 'logo', 'favicon', 'avatar', 'button']):
                    image_url = result['thumbnail']
                    print(f"Selected thumbnail URL: {image_url} (score: {result.get('image_score', 0)})")
                    print(f"Image source: {result.get('url', 'unknown')}")
                    print(f"Image title: {result.get('title', 'unknown')}")
                    break
            elif result.get('imageUrl') and isinstance(result.get('imageUrl'), str) and result.get('imageUrl').startswith(('http://', 'https://')):
                # Verify the imageUrl doesn't contain terms that suggest it's an icon or logo
                img_url_lower = result.get('imageUrl').lower()
                if not any(term in img_url_lower for term in ['icon', 'logo', 'favicon', 'avatar', 'button']):
                    image_url = result['imageUrl']
                    print(f"Selected imageUrl: {image_url} (score: {result.get('image_score', 0)})")
                    print(f"Image source: {result.get('url', 'unknown')}")
                    print(f"Image title: {result.get('title', 'unknown')}")
                    break
                    
        # Second pass: If no high-quality images found, accept any image including favicons
        if not image_url:
            for result in prioritized_results:
                if isinstance(result.get('image'), str) and result.get('image').startswith(('http://', 'https://')):
                    image_url = result['image']
                    print(f"Selected fallback image URL: {image_url} (score: {result.get('image_score', 0)})")
                    print(f"Image source: {result.get('url', 'unknown')}")
                    print(f"Image title: {result.get('title', 'unknown')}")
                    break
                elif result.get('thumbnail') and isinstance(result.get('thumbnail'), str) and result.get('thumbnail').startswith(('http://', 'https://')):
                    image_url = result['thumbnail']
                    print(f"Selected fallback thumbnail URL: {image_url} (score: {result.get('image_score', 0)})")
                    print(f"Image source: {result.get('url', 'unknown')}")
                    print(f"Image title: {result.get('title', 'unknown')}")
                    break
                elif result.get('imageUrl') and isinstance(result.get('imageUrl'), str) and result.get('imageUrl').startswith(('http://', 'https://')):
                    image_url = result['imageUrl']
                    print(f"Selected fallback imageUrl: {image_url} (score: {result.get('image_score', 0)})")
                    print(f"Image source: {result.get('url', 'unknown')}")
                    print(f"Image title: {result.get('title', 'unknown')}")
                    break
                elif isinstance(result.get('image'), bool) and result.get('favicon') and isinstance(result.get('favicon'), str) and result.get('favicon').startswith(('http://', 'https://')):
                    # Only use favicon as a last resort
                    image_url = result['favicon']
                    print(f"Selected favicon as image URL: {image_url} (score: {result.get('image_score', 0)})")
                    print(f"Image source: {result.get('url', 'unknown')}")
                    print(f"Image title: {result.get('title', 'unknown')}")
                    break

        if not image_url:
            print("No suitable image URL found in any results")
            # If we couldn't find any image, try a fallback search specifically for images
            if hasattr(self, 'current_topic') and self.current_topic:
                print(f"Attempting fallback image search for topic: {self.current_topic}")
                
                # Perform a dedicated image search with broader terms
                fallback_query = f"{self.current_topic} educational diagram illustration image"
                try:
                    fallback_results = self._search_web(fallback_query, educational_focus=True)
                    
                    # Try to find any image in the fallback results
                    for result in fallback_results:
                        if isinstance(result.get('image'), str) and result.get('image').startswith(('http://', 'https://')):
                            image_url = result['image']
                            print(f"Found fallback image URL: {image_url}")
                            break
                        elif result.get('thumbnail') and isinstance(result.get('thumbnail'), str) and result.get('thumbnail').startswith(('http://', 'https://')):
                            image_url = result['thumbnail']
                            print(f"Found fallback thumbnail URL: {image_url}")
                            break
                except Exception as e:
                    print(f"Error in fallback image search: {e}")
                    
            # If still no image, use a default educational image based on topic category
            if not image_url:
                # Categorize the topic to find an appropriate default image
                topic_categories = {
                    'science': ['physics', 'chemistry', 'biology', 'astronomy', 'space', 'earth', 'environment', 'technology', 'engineering'],
                    'math': ['mathematics', 'algebra', 'geometry', 'calculus', 'statistics', 'arithmetic', 'number', 'equation'],
                    'history': ['history', 'civilization', 'war', 'ancient', 'medieval', 'modern', 'revolution', 'empire', 'kingdom'],
                    'geography': ['geography', 'map', 'country', 'continent', 'ocean', 'river', 'mountain', 'climate', 'weather'],
                    'literature': ['literature', 'book', 'novel', 'poem', 'author', 'writer', 'story', 'character', 'fiction'],
                    'art': ['art', 'painting', 'sculpture', 'music', 'dance', 'theater', 'film', 'photography', 'design'],
                }
                
                # Default educational images by category
                default_images = {
                    'science': 'https://cdn.pixabay.com/photo/2018/09/27/09/22/artificial-intelligence-3706562_1280.jpg',
                    'math': 'https://cdn.pixabay.com/photo/2015/11/15/07/47/geometry-1044090_1280.jpg',
                    'history': 'https://cdn.pixabay.com/photo/2018/08/15/07/19/indian-history-3607311_1280.jpg',
                    'geography': 'https://cdn.pixabay.com/photo/2016/10/30/20/22/map-1784029_1280.jpg',
                    'literature': 'https://cdn.pixabay.com/photo/2015/11/19/21/10/glasses-1052010_1280.jpg',
                    'art': 'https://cdn.pixabay.com/photo/2016/06/25/12/55/art-1478831_1280.jpg',
                    'default': 'https://cdn.pixabay.com/photo/2015/07/31/11/45/library-869061_1280.jpg'  # General education image
                }
                
                # Determine the category of the topic
                if hasattr(self, 'current_topic'):
                    topic_lower = self.current_topic.lower()
                    matched_category = 'default'
                    
                    for category, keywords in topic_categories.items():
                        if any(keyword in topic_lower for keyword in keywords):
                            matched_category = category
                            break
                    
                    # Use the default image for the matched category
                    image_url = default_images.get(matched_category, default_images['default'])
                    print(f"Using default {matched_category} image: {image_url}")
                else:
                    # If no topic is set, use the general education image
                    image_url = default_images['default']
                    print(f"Using general default image: {image_url}")

        formatted_results = []
        
        # Add results by category
        for category, results in categories.items():
            if results:
                formatted_results.append(f"\n{category} Resources:")
                for result in results:
                    formatted_results.append(self._format_web_resource(result))

        if not formatted_results:  # If no results found
            formatted_text = "No additional web resources found."
        else:
            formatted_text = "\n".join(formatted_results[:5])  # Limit to 5 lines

        return {
            "formatted_text": formatted_text,
            "image_url": image_url
        }

    def _format_web_resource(self, result: Dict[str, Any]) -> str:
        """
        Format a single web resource with its type.
        """
        url = result.get('url', '')
        domain = url.split('/')[2] if '/' in url else url

        if 'youtube.com' in domain or 'youtu.be' in domain:
            resource_type = "YouTube Video"
        elif 'twitter.com' in domain or 'x.com' in domain:
            resource_type = "Twitter/X Post"
        elif 'wikipedia.org' in domain:
            resource_type = "Wikipedia Article"
        elif any(site in domain for site in ['news', 'times', 'post', 'journal', 'reuters', 'bbc', 'cnn']):
            resource_type = "News Article"
        elif any(site in domain for site in ['edu', 'org', 'gov']):
            resource_type = "Educational Resource"
        else:
            resource_type = "Web Resource"

        title = result.get('title', 'No title')
        summary = result.get('summary', '')
        highlights = result.get('highlights', [])

        formatted = [f"- [{title}]({url}) - {resource_type}"]
        if summary:
            formatted.append(f"  - Summary: {summary[:50]}...")
        if highlights:
            formatted.append("  - Highlights:")
            for highlight in highlights[:2]:
                formatted.append(f"    - {highlight[:50]}...")
        return "\n".join(formatted)

    def reset_conversation(self):
        """
        Reset the conversation context and topic history.
        """
        # Reset LlamaIndex memory buffer
        self.memory.reset()
        
        # Reset manual conversation history
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Clear topic and interest tracking
        self.topic_history = []
        self.subtopic_history = []
        self.user_interests = {}
        self.interest_categories = {}

    def _manage_topic_history(self, topic: str):
        """
        Manage the topic history and update user interests based on explored topics.
        Ensures proper connections between topics in the learning journey.
        """
        # Normalize topic for case-insensitive comparison
        normalized_topic = topic.lower().strip()
        
        # Check if topic is already in history (case-insensitive)
        existing_topics = [t.lower().strip() for t in self.topic_history]
        if normalized_topic not in existing_topics:
            # Add new topic to history
            self.topic_history.append(topic)
            
            # Limit history size
            if len(self.topic_history) > self.max_history:
                self.topic_history.pop(0)
            
            # Update user interests with the new topic
            self._update_user_interests(topic)
        else:
            # If topic exists but in different case/format, update to the new format
            # but don't change its position in history (to maintain chronology)
            for i, t in enumerate(self.topic_history):
                if t.lower().strip() == normalized_topic:
                    self.topic_history[i] = topic
                    break
            
            # Still update interests to reinforce this topic
            self._update_user_interests(topic)

    def _update_user_interests(self, topic: str):
        """Update user interests based on a topic."""
        # Normalize topic
        topic = topic.strip()
        
        # Increment interest in this topic
        if topic in self.user_interests:
            self.user_interests[topic] += 1
        else:
            self.user_interests[topic] = 1
            
        # Categorize the topic if not already done
        if topic not in self.interest_categories:
            self._categorize_topic(topic)
            
    def _categorize_topic(self, topic: str):
        """Categorize a topic into broader subject areas."""
        try:
            # Use a simple keyword matching approach for now
            topic_lower = topic.lower()
            
            categories = {
                "math": ["math", "algebra", "geometry", "calculus", "statistics", "number"],
                "science": ["science", "physics", "chemistry", "biology", "astronomy", "earth"],
                "history": ["history", "ancient", "medieval", "modern", "civilization", "war"],
                "literature": ["literature", "poetry", "novel", "fiction", "writing", "author"],
                "arts": ["art", "music", "painting", "sculpture", "dance", "theater"],
                "technology": ["technology", "computer", "programming", "digital", "internet", "ai"],
                "social_studies": ["social", "geography", "economics", "politics", "psychology", "sociology"]
            }
            
            for category, keywords in categories.items():
                if any(keyword in topic_lower for keyword in keywords):
                    self.interest_categories[topic] = category
                    return
                    
            # If no category matched, use "general"
            self.interest_categories[topic] = "general"
        except Exception as e:
            print(f"Error categorizing topic: {e}")
        
        if matched_fields:
            return matched_fields[0]
        else:
            # If no match found, return a random educational field
            import random
            random_field = random.choice(list(common_fields.keys()))
            return random.choice(common_fields[random_field])

    def _update_user_interests(self, topic: str):
        """
        Update the user interest model based on the explored topic.
        Analyzes the topic and categorizes it to build a user interest profile.
        """
        # Categorize the topic if not already categorized
        if topic not in self.interest_categories:
            # Use LLM to categorize the topic into broader subject areas
            prompt = f"""
            Categorize the educational topic "{topic}" into 2-3 broader subject areas or categories.
            For example, "Black Holes" might be categorized as "Astronomy", "Physics", and "Space Science".
            Return only a JSON array of category strings, nothing else.
            """
            try:
                response = self._get_completion_with_history(prompt, temperature=0.3)
                categories = json.loads(response)
                if isinstance(categories, list):
                    self.interest_categories[topic] = categories
                    
                    # Update weights in user interests
                    for category in categories:
                        if category in self.user_interests:
                            self.user_interests[category] += 1
                        else:
                            self.user_interests[category] = 1
            except Exception as e:
                print(f"Error categorizing topic: {e}")
                # Fallback: use the topic itself as a category
                self.interest_categories[topic] = [topic]
                if topic in self.user_interests:
                    self.user_interests[topic] += 1
                else:
                    self.user_interests[topic] = 1
    
    def _infer_learning_focus(self, topics):
        """
        Analyze a list of topics to infer the user's learning focus or educational journey.
        """
        if not topics or len(topics) < 2:
            return "exploring foundational concepts"
            
        # Use a simple approach to categorize the learning journey
        # For more sophisticated analysis, this could be enhanced with LLM calls
        common_subjects = {
            "science": ["physics", "chemistry", "biology", "astronomy", "earth", "space", "solar", "planet", "element", "atom", "molecule"],
            "math": ["algebra", "geometry", "calculus", "statistics", "number", "equation", "function", "graph"],
            "history": ["ancient", "medieval", "modern", "world", "war", "civilization", "empire", "revolution"],
            "literature": ["poetry", "novel", "fiction", "drama", "author", "character", "plot", "theme"],
            "geography": ["continent", "country", "map", "climate", "mountain", "river", "ocean", "population"],
            "technology": ["computer", "internet", "digital", "programming", "software", "hardware", "algorithm"],
        }
        
        # Count subject area matches
        subject_counts = {subject: 0 for subject in common_subjects}
        for topic in topics:
            topic_lower = topic.lower()
            for subject, keywords in common_subjects.items():
                if any(keyword in topic_lower for keyword in keywords):
                    subject_counts[subject] += 1
        
        # Find the dominant subject area
        dominant_subject = max(subject_counts.items(), key=lambda x: x[1])
        if dominant_subject[1] > 0:
            return f"understanding {dominant_subject[0]}"
            
        # If no clear subject area, analyze the progression pattern
        if len(topics) >= 3:
            # Check if moving from general to specific
            if len(topics[0]) < len(topics[-1]) and any(t in topics[-1].lower() for t in topics[0].lower().split()):
                return "exploring topics in increasing depth and specificity"
            # Check if exploring related concepts
            elif any(topics[-1].lower() in t.lower() or t.lower() in topics[-1].lower() for t in topics[:-1]):
                return "exploring related concepts in a connected field"
        
        return "building a comprehensive understanding across multiple topics"
    
    def _get_user_interest_context(self):
        """
        Generate a context string based on user interests to guide topic recommendations.
        """
        if not self.user_interests:
            return ""
            
        # Sort interests by weight
        sorted_interests = sorted(self.user_interests.items(), key=lambda x: x[1], reverse=True)
        top_interests = sorted_interests[:5]  # Take top 5 interests
        
        interest_context = "Based on the user's exploration history, they appear interested in: " + \
                          ", ".join([f"{interest} (weight: {weight})" for interest, weight in top_interests])
        
        # Add recent topics
        if self.topic_history:
            recent_topics = self.topic_history[-3:]  # Last 3 topics
            interest_context += f". Their recent topics include: {', '.join(recent_topics)}."
            
        return interest_context


# Example usage:
if __name__ == "__main__":
    bot = CuriosityBlocksAPI()
    topics = bot.generate_topics("4", "CBSE")
    print("Generated Topics:", topics)