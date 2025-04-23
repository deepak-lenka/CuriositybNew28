import streamlit as st
from curiosity_blocks_api import CuriosityBlocksAPI
import json

# Initialize API
api = CuriosityBlocksAPI()

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
    
    if st.button("Generate Topics"):
        with st.spinner("Generating topics..."):
            topics = api.generate_topics(grade, board)
            if topics:
                st.subheader("Suggested Topics")
                for topic in topics:
                    st.markdown(f"- **{topic['topic']}**: {topic['description']}")

    # Topic exploration section
    st.header("Explore Topic")
    topic = st.text_input("Enter a topic to explore")
    
    if st.button("Explore Topic") and topic:
        with st.spinner("Generating explanation..."):
            content = api.explain_topic(topic, grade, board)
            
            if content:
                # Show image if available (before title)
                image_url = content["main_topic"].get("image_url")
                if image_url:
                    try:
                        st.image(image_url, caption=content["main_topic"]["title"], use_container_width=True)
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

                # Display related topics with web resources
                if content["related_topics"]:
                    st.subheader("Related Topics")
                    for related in content["related_topics"]:
                        st.markdown(f"- **{related['topic']}**: {related['summary']}")
                        if related.get("web_resources"):
                            st.markdown(f"  Web Resources: {related['web_resources']}")

    # Related topics section
    st.header("Get Related Topics")
    if st.button("Get Related Topics") and topic:
        with st.spinner("Finding related topics..."):
            related_topics = api.explore_related_topics(topic, grade, board)
            if related_topics:
                st.subheader("Related Topics")
                for topic in related_topics:
                    st.markdown(f"- **{topic['topic']}**: {topic['summary']}")
                    if topic.get("web_resources"):
                        st.markdown(f"  Web Resources: {topic['web_resources']}")

    # Reset conversation
    if st.button("Reset Conversation"):
        api.reset_conversation()
        st.success("Conversation history has been cleared!")

if __name__ == "__main__":
    main()
