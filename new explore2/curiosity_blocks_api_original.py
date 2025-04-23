import os
import json
import requests
from typing import List, Dict, Any
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

class CuriosityBlocksAPI:
    def __init__(self):
        """
        Initialize the CuriosityBlocksAPI with LangChain components.
        """
        # Use ConversationBufferWindowMemory to limit conversation history
        self.memory = ConversationBufferWindowMemory(
            return_messages=True,
            k=3  # Keep only last 3 interactions
        )
        self.llm = ChatOpenAI(
            model_name="gpt-3.5-turbo-16k",  # Use 16k context model
            temperature=0.7,
            max_tokens=4000,  # Limit response length
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Create prompt template with memory
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an educational assistant for students. Be helpful, educational, and age-appropriate."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        # Initialize conversation chain
        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=self.prompt,
            verbose=False
        )
        self.topic_history = []
        self.max_history = 5  # Keep only last 5 topics
        self.exa_api_key = os.getenv("EXA_API_KEY")
        self.exa_base_url = "https://api.exa.ai/search"
        print(f"Exa API Key loaded: {'***' if self.exa_api_key else 'NO API KEY'}")  # Debug print

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
        
        # Configure search to include images
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
        
        # Determine resource type
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
        
        formatted = []
        formatted.append(f"- [{title}]({url}) - {resource_type}")
        if summary:
            formatted.append(f"  - Summary: {summary[:50]}...")  # Limit summary to 50 chars
        if highlights:
            formatted.append("  - Highlights:")
            for highlight in highlights[:2]:
                formatted.append(f"    - {highlight[:50]}...")  # Limit highlights to 50 chars

        return "\n".join(formatted)

    def _get_completion_with_history(self, prompt: str) -> str:
        """
        Get a completion using LangChain's conversation memory.
        """
        try:
            # Add limited context from topic history
            context = ""
            if self.topic_history:
                recent_topics = self.topic_history[-3:]  # Only use last 3 topics
                context = f"Recent topics explored: {', '.join(recent_topics)}. "

            response = self.conversation.predict(input=context + prompt)
            return response
        except Exception as e:
            print(f"Error with LangChain completion: {e}")
            return None

    def generate_topics(self, grade: str, board: str, n_topics: int = 4) -> List[Dict[str, str]]:
        """
        Generate curiosity topics based on grade level and board.
        """
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
            return topics
        except Exception as e:
            print(f"Error parsing topics JSON: {e}")
            return []

    def explain_topic(self, topic: str, grade: str, board: str) -> Dict[str, Any]:
        """
        Generate an engaging explanation for a selected topic.
        """
        self._manage_topic_history(topic)

        # First get the explanation from OpenAI
        prompt = f"""
        Generate a concise educational explanation about "{topic}" for {grade} grade students.
        The explanation should be at least 100 words and complete sentences.
        Format your response as a JSON object with this exact structure:
        {{
            "main_topic": {{
                "title": "{topic}",
                "explanation": "At least 100 words explanation with complete sentences"
            }},
            "subtopics": [
                {{
                    "title": "subtopic name",
                    "explanation": "1-2 sentence explanation"
                }},
                // 2-3 more subtopics with same structure
            ],
            "related_topics": [
                {{
                    "topic": "related topic name",
                    "summary": "2-3 sentence explanation of why this topic is relevant"
                }},
                // 2 more related topics with same structure
            ]
        }}
        Ensure all fields match these exact names and structure.
        The main topic explanation must be at least 100 words and end with a complete sentence.
        """

        response = self._get_completion_with_history(prompt)
        try:
            content = json.loads(response)
            
            # Process main topic explanation
            main_topic = content["main_topic"]
            explanation = main_topic["explanation"]
            
            # Ensure explanation is at least 100 words and ends with a complete sentence
            words = explanation.split()
            if len(words) < 100:
                # If less than 100 words, keep adding words until we reach at least 100
                # or until we reach the end of a complete sentence
                current_length = len(words)
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

            # Get web resources for main topic
            main_topic_query = f"{main_topic['title']} {main_topic['explanation']} educational content"
            main_topic_results = self._search_web(main_topic_query)
            web_resources = self._format_web_results(main_topic_results)
            main_topic['web_resources'] = web_resources['formatted_text']
            main_topic['image_url'] = web_resources['image_url']

            # Get web resources for subtopics
            for subtopic in content["subtopics"]:
                subtopic_query = f"{subtopic['title']} {subtopic['explanation']} educational content"
                subtopic_results = self._search_web(subtopic_query)
                subtopic['web_resources'] = self._format_web_results(subtopic_results)['formatted_text']

            return {
                "main_topic": main_topic,
                "subtopics": content["subtopics"],
                "related_topics": content["related_topics"]
            }

        except Exception as e:
            print(f"Error parsing explanation JSON: {e}")
            return {
                "main_topic": {"title": topic, "explanation": "Failed to generate explanation"},
                "subtopics": [],
                "related_topics": []
            }

    def explore_related_topics(self, topic: str, grade: str, board: str) -> List[Dict[str, str]]:
        """
        Generate related topics based on the current topic being explored.
        """
        self._manage_topic_history(topic)

        prompt = f"""
        Based on the topic "{topic}", suggest exactly 3 related topics for {grade} grade students.
        For each topic, provide a clear and engaging title and a concise summary that explains why it's relevant.
        The summary should be exactly 2-3 sentences long and should help students understand the connection to the main topic.
        Format as a JSON array with exactly 3 objects, each containing:
        - topic (string): Related topic name (max 50 chars)
        - summary (string): A concise 2-3 sentence explanation of why this topic is relevant
        """

        response = self._get_completion_with_history(prompt)
        try:
            related_topics = json.loads(response)
            
            # Ensure exactly 3 topics
            if len(related_topics) > 3:
                related_topics = related_topics[:3]
            elif len(related_topics) < 3:
                # Add placeholder topics if needed
                while len(related_topics) < 3:
                    related_topics.append({
                        "topic": f"Additional Topic {len(related_topics) + 1}",
                        "summary": "This is a placeholder topic that will be replaced with actual content."
                    })

            return related_topics
        except Exception as e:
            print(f"Error parsing related topics JSON: {e}")
            return []

    def reset_conversation(self):
        """
        Reset the conversation context and topic history.
        """
        self.memory.clear()
        self.topic_history = []

    def _manage_topic_history(self, topic: str):
        """
        Manage topic history to prevent it from growing too large.
        """
        if topic not in self.topic_history:
            if len(self.topic_history) >= self.max_history:
                self.topic_history.pop(0)  # Remove oldest topic
            self.topic_history.append(topic)
