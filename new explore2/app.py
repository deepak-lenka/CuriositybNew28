import streamlit as st
from curiosity_blocks_api import CuriosityBlocksAPI
import json

# Initialize API
api = CuriosityBlocksAPI()

# Initialize session state variables
if 'current_topic' not in st.session_state:
    st.session_state.current_topic = ""
if 'explore_clicked' not in st.session_state:
    st.session_state.explore_clicked = False
if 'generate_topics_clicked' not in st.session_state:
    st.session_state.generate_topics_clicked = False
if 'get_related_clicked' not in st.session_state:
    st.session_state.get_related_clicked = False
if 'generated_topics' not in st.session_state:
    st.session_state.generated_topics = []
if 'related_topics' not in st.session_state:
    st.session_state.related_topics = []
if 'topic_content' not in st.session_state:
    st.session_state.topic_content = None

# Callback functions for buttons
def explore_topic_callback():
    st.session_state.explore_clicked = True

def generate_topics_callback():
    st.session_state.generate_topics_clicked = True

def get_related_callback():
    st.session_state.get_related_clicked = True
    # Prevent re-exploration of the main topic
    st.session_state.explore_clicked = False

def explore_specific_topic(topic_name):
    st.session_state.current_topic = topic_name
    st.session_state.explore_clicked = True

def main():
    st.title("Curiosity Blocks - Educational Topic Explorer")
    
    # Sidebar for grade and board selection
    st.sidebar.header("Student Profile")
    grade = st.sidebar.selectbox(
        "Select Grade",
        ["5", "6", "7", "8", "9", "10", "11", "12"]
    )
    board = st.sidebar.selectbox(
        "Select Board",
        ["CBSE", "ICSE", "State Board"]
    )

    # Main content
    st.header("Generate Topics")
    
    if st.button("Generate Topics", on_click=generate_topics_callback):
        pass
    
    # Process generate topics request
    if st.session_state.generate_topics_clicked:
        with st.spinner("Generating topics..."):
            topics = api.generate_topics(grade, board)
            st.session_state.generated_topics = topics
            st.session_state.generate_topics_clicked = False
    
    # Display generated topics
    if st.session_state.generated_topics:
        st.subheader("Suggested Topics")
        for i, topic in enumerate(st.session_state.generated_topics):
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button(f"Explore", key=f"gen_topic_{i}"):
                    explore_specific_topic(topic['topic'])
            with col2:
                st.markdown(f"**{topic['topic']}**: {topic['description']}")

    # Topic exploration section
    st.header("Explore Topic")
    
    # Show previous topics for context if available
    if hasattr(api, 'topic_history') and api.topic_history and len(api.topic_history) > 0:
        st.caption(f"Previously explored: {', '.join(api.topic_history[-3:])}")
    
    topic = st.text_input("Enter a topic to explore", value=st.session_state.current_topic)
    
    if st.button("Explore Topic") and topic:
        # Directly update the session state with the manually entered topic
        st.session_state.current_topic = topic
        st.session_state.explore_clicked = True
        # Clear any previous topic content to ensure fresh content is generated
        st.session_state.topic_content = None
    
    # Process explore topic request
    if st.session_state.explore_clicked and st.session_state.current_topic and not st.session_state.get_related_clicked:
        with st.spinner("Generating explanation..."):
            content = api.explain_topic(st.session_state.current_topic, grade, board)
            st.session_state.topic_content = content
            st.session_state.explore_clicked = False
    
    # Display topic content
    if st.session_state.topic_content:
        content = st.session_state.topic_content
        
        # Show image if available
        image_url = content["main_topic"].get("image_url")
        if image_url:
            try:
                # Validate the image URL before trying to display it
                if isinstance(image_url, str) and (image_url.startswith('http://') or image_url.startswith('https://')):
                    st.image(image_url, caption=content["main_topic"]["title"], use_container_width=True)
                else:
                    print(f"Invalid image URL format: {image_url}")
                    st.text("Image could not be loaded - invalid URL format")
            except Exception as e:
                print(f"Error displaying image: {e}")
                st.text("Image could not be loaded")
        
        # Display main topic title and explanation
        st.subheader(content["main_topic"]["title"])
        st.markdown(content["main_topic"]["explanation"])

        # Display subtopics with web resources
        if content["subtopics"]:
            st.subheader("Subtopics")
            for subtopic in content["subtopics"]:
                st.markdown(f"- **{subtopic['title']}**: {subtopic['explanation']}")
                if subtopic.get("web_resources"):
                    st.markdown(f"  Web Resources: {subtopic['web_resources']}")

        # Display related topics with web resources directly from the main topic content
        if content["related_topics"]:
            st.subheader("Related Topics")
            for i, related in enumerate(content["related_topics"]):
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.button(f"Explore", key=f"content_related_{i}"):
                        explore_specific_topic(related['topic'])
                with col2:
                    st.markdown(f"**{related['topic']}**: {related['summary']}")
                    # Removed web resources from related topics as requested
        
        # Only show the "Get More Related Topics" button if a topic has been explored
        st.header("Get More Related Topics")
        if st.button("Get More Related Topics", key="more_related_topics", on_click=get_related_callback):
            # This prevents the main topic from being re-explored
            pass
    
    # Process get related topics request - only if a topic has been explored
    if st.session_state.get_related_clicked and st.session_state.current_topic:
        with st.spinner("Finding more related topics..."):
            related_topics = api.explore_related_topics(st.session_state.current_topic, grade, board)
            st.session_state.related_topics = related_topics
            st.session_state.get_related_clicked = False
    
    # Display additional related topics only if they exist and a main topic has been explored
    if st.session_state.related_topics and st.session_state.topic_content:
        st.subheader("Additional Related Topics")
        for i, related_topic in enumerate(st.session_state.related_topics):
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button(f"Explore", key=f"related_{i}"):
                    explore_specific_topic(related_topic['topic'])
            with col2:
                st.markdown(f"**{related_topic['topic']}**: {related_topic['description']}")

    # Reset conversation
    if st.button("Reset Conversation"):
        api.reset_conversation()
        # Clear all session state variables
        st.session_state.current_topic = ""
        st.session_state.explore_clicked = False
        st.session_state.generate_topics_clicked = False
        st.session_state.get_related_clicked = False
        st.session_state.generated_topics = []
        st.session_state.related_topics = []
        st.session_state.topic_content = None
        st.success("Conversation history has been cleared!")

if __name__ == "__main__":
    main()
