import os
import json
import requests
import openai
from typing import List, Dict, Any
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex


# Set the API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

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
                if os.path.exists(data_dir):
                    documents = SimpleDirectoryReader(data_dir).load_data()
                    self.index = VectorStoreIndex.from_documents(documents)
                    self.retriever = self.index.as_retriever()
                else:
                    print(f"Data directory not found: {data_dir}")
                    self.retriever = None
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
        """
        # Always use the direct OpenAI API approach since we're having issues with the retriever
        try:
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo-16k",
                messages=self.conversation_history + [{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            reply = response.choices[0].message.content
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            print(f"Error in OpenAI API call: {str(e)}")
            return "I'm having trouble processing that right now. Let's try something else."

    def generate_topics(self, grade: str, board: str, n_topics: int = 4) -> List[Dict[str, str]]:
        """
        Generate curiosity topics based on grade level and board.
        """
        try:
            prompt = f"""
            Generate {n_topics} interesting educational topics for {grade} grade students in the {board} curriculum.
            Keep responses concise and focused.
            Format the response as a JSON array with {n_topics} topic objects, each containing:
            - topic (string): The name of the topic (max 50 chars)
            - description (string): A brief 1-sentence description (max 100 chars)
            """
            response = self._get_completion_with_history(prompt)
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

    def explain_topic(self, topic: str, grade: str, board: str):
        """
        Generate an engaging explanation for a selected topic.
        Takes into account user's interests and exploration history.
        """
        # Update topic history and user interests
        try:
            self._manage_topic_history(topic)
            interest_context = self._get_user_interest_context()
        except Exception as e:
            print(f"Error preparing topic context: {e}")
            interest_context = ""
        
        # Generate the explanation
        try:
            # Prepare prompt for generating content with personalization
            prompt = f"""
            Create an educational explanation about "{topic}" for {grade} grade students following the {board} curriculum.
            The explanation should be engaging, informative, and appropriate for their grade level.
            
            {interest_context}
            
            Please tailor the explanation to connect with the user's interests where relevant, while maintaining educational accuracy.
            
            Structure the response as a JSON object with the following sections:
            1. main_topic: An object containing:
               - title: A catchy title for the topic
               - explanation: A detailed, engaging explanation (300-500 words) that connects to the user's interests where relevant
               - image_url: (optional) A URL to a relevant educational image
            
            2. subtopics: An array of 2-3 objects, each containing:
               - title: A clear title for the subtopic
               - explanation: A concise explanation (100-150 words)
               - web_resources: (optional) Links to educational resources
            
            3. related_topics: An array of 2-3 objects, each containing:
               - topic: Name of a related topic that connects to both the main topic and the user's interests
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
                # Add placeholder topics if needed
                while len(related_topics) < 3:
                    related_topics.append({
                        "topic": f"Additional Topic {len(related_topics) + 1}",
                        "summary": "This is a placeholder topic that will be replaced with actual content."
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

            # Get web resources for main topic to find an image
            try:
                main_topic_query = f"{main_topic['title']} {main_topic['explanation']} educational content"
                main_topic_results = self._search_web(main_topic_query)
                web_resources = self._format_web_results(main_topic_results)
                main_topic['web_resources'] = web_resources['formatted_text']
                main_topic['image_url'] = web_resources['image_url']
            except Exception as e:
                print(f"Error getting web resources for main topic: {e}")
                main_topic['web_resources'] = ""
                main_topic['image_url'] = ""

            # Get web resources for subtopics
            for subtopic in content["subtopics"]:
                try:
                    subtopic_query = f"{subtopic['title']} {subtopic['explanation']} educational content"
                    subtopic_results = self._search_web(subtopic_query)
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
            
            # Build a more personalized prompt based on user interests
            prompt = f"""
            Based on the topic "{topic}", suggest exactly 3 related topics for {grade} grade students.
            
            {interest_context}
            
            Please consider the user's interests when suggesting related topics, but ensure they are still 
            relevant to the main topic "{topic}". The topics should be educational and appropriate for {grade} grade students.
            
            For each topic, provide:
            1. A clear and engaging title
            2. A concise summary that explains why it's relevant (2-3 sentences)
            3. A brief explanation of how it connects to the user's previous interests, if applicable
            
            Format as a JSON array with exactly 3 objects, each containing:
            - topic (string): Related topic name (max 50 chars)
            - description (string): A concise explanation of why this topic is relevant
            - connection (string): How this connects to the user's interests (if applicable)
            """
            response = self._get_completion_with_history(prompt)
            try:
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
                            description += f" {connection}"
                            
                        formatted_topics.append({
                            "topic": topic if topic else f"Related Topic {len(formatted_topics) + 1}",
                            "description": description
                        })
                
                # Ensure exactly 3 topics
                while len(formatted_topics) < 3:
                    formatted_topics.append({
                        "topic": f"Related Topic {len(formatted_topics) + 1}",
                        "description": "This is a placeholder topic that will be replaced with actual content."
                    })
                
                return formatted_topics[:3]  # Return exactly 3 topics
            except json.JSONDecodeError as e:
                print(f"Error parsing related topics JSON: {e}")
                print(f"Raw response: {response}")
                # Return a fallback response
                return [
                    {"topic": f"Related Topic {i+1}", "description": "Related topic description"}
                    for i in range(3)
                ]
        except Exception as e:
            print(f"Error in explore_related_topics: {e}")
            raise

    def _search_web(self, query: str) -> List[Dict[str, Any]]:
        """
        Search the web using Exa AI's search API.
        """
        if not self.exa_api_key:
            print("No EXA_API_KEY found in environment variables")
            return []

        headers = {
            "Authorization": f"Bearer {self.exa_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "query": query,
            "type": "auto",
            "numResults": 10,
            "contents": {
                "text": True,
                "highlights": True,
                "summary": True,
                "image": True,  # Explicitly request image content
                "livecrawl": "always",
                "livecrawlTimeout": 5000
            }
        }

        try:
            print(f"Making Exa API request for query: {query}")
            response = requests.post(
                self.exa_base_url,
                headers=headers,
                json=payload,
                timeout=15
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
                        has_image = bool(result.get('image') or result.get('thumbnail') or result.get('imageUrl'))
                        print(f"Result has image: {has_image}")
                        filtered_results.append(result)
                
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
            return {
                "formatted_text": "No additional web resources found.",
                "image_url": None
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

        # Try to find an image URL from the results
        image_url = None
        for result in results:
            # Check for image URL in various fields
            if result.get('image'):
                image_url = result['image']
                print(f"Found image URL: {image_url}")
                break
            elif result.get('thumbnail'):
                image_url = result['thumbnail']
                print(f"Found thumbnail URL: {image_url}")
                break
            elif result.get('imageUrl'):
                image_url = result['imageUrl']
                print(f"Found imageUrl: {image_url}")
                break

        if not image_url:
            print("No image URL found in any results")

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
        """
        # Add to topic history if not already the most recent topic
        if not self.topic_history or self.topic_history[-1] != topic:
            self.topic_history.append(topic)
            if len(self.topic_history) > self.max_history:
                self.topic_history.pop(0)
            
            # Update user interests
            self._update_user_interests(topic)
    
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