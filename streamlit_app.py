import os
os.environ["STREAMLIT_WATCHER_DISABLE_AUTO_WATCH"] = "true"
import streamlit as st
import sys

# Ensure local module import works
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rag_chatbot_main import EnhancedFreeRAGChatbot

st.set_page_config(page_title="Neurific AI RAG Chatbot", page_icon="🤖", layout="wide")

# Custom CSS for better LaTeX rendering
st.markdown("""
<style>
.stMarkdown {
    font-family: 'Computer Modern', serif;
}
.math {
    font-size: 1.1em;
}
</style>
""", unsafe_allow_html=True)

st.title("🤖 Neurific AI RAG Chatbot")
st.markdown("""
Welcome! This chatbot answers **LSTM** questions with authoritative sources and beautiful mathematical formatting.

**Sources:** Chris Olah's LSTM Blog, CMU LSTM Notes  
**Backend:** 100% free, local, and private  
""")

@st.cache_resource(show_spinner="Loading the chatbot and indexing sources. This may take a minute on first run…")
def get_chatbot():
    bot = EnhancedFreeRAGChatbot()
    bot.initialize()
    return bot

chatbot = get_chatbot()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # Use st.markdown for better formatting (including LaTeX)
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            if message["sources"]:
                st.markdown("**Sources Used:**")
                for src in message["sources"]:
                    st.markdown(f"- **{src['source_name']}**")

# Quick action buttons
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📐 Show LSTM Formulas"):
        st.session_state.messages.append({"role": "user", "content": "mathematical formulation of LSTM"})
        st.rerun()
with col2:
    if st.button("🧠 How do LSTMs work?"):
        st.session_state.messages.append({"role": "user", "content": "How do LSTMs work and what makes them special?"})
        st.rerun()
with col3:
    if st.button("🔄 Vanishing Gradient"):
        st.session_state.messages.append({"role": "user", "content": "What is the vanishing gradient problem and how do LSTMs solve it?"})
        st.rerun()

# Accept user input using Streamlit's chat input
if prompt := st.chat_input("Ask your LSTM question (try 'mathematical formulation of LSTM')"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get chatbot response
    with st.chat_message("assistant"):
        with st.spinner("Generating answer..."):
            response = chatbot.ask_question(prompt)
            
            # Display the answer as markdown (supports LaTeX)
            st.markdown(response["answer"])
            
            # Show quality score and response time
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Quality Score", f"{response['quality_score']*10:.1f}/10")
            with col2:
                st.metric("Response Time", f"{response.get('response_time', 0):.2f}s")
            
            # Optionally, show sources in a nice way
            if response.get("sources"):
                with st.expander("📚 View Sources"):
                    for src in response["sources"]:
                        st.markdown(f"**{src['source_name']}** (Weight: {src['quality_weight']})")
                        st.caption(f"Preview: {src['content_preview']}")
            
            # Add assistant message to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response["answer"],
                "sources": response.get("sources", [])
            })

# Sidebar with stats and about info
with st.sidebar:
    st.header("About")
    st.info(
        "This chatbot uses Retrieval-Augmented Generation (RAG) with local LLMs and open-source embeddings. "
        "All answers are sourced from Chris Olah's blog and CMU LSTM notes.\n\n"
        "**Special Feature:** Beautiful LaTeX rendering for mathematical formulas!\n\n"
        "No data leaves your computer."
    )
    
    if st.button("📊 Show Performance Stats"):
        stats = chatbot.get_performance_stats()
        st.write("**Performance Stats:**")
        for k, v in stats.items():
            if isinstance(v, float):
                st.write(f"- {k}: {v:.3f}")
            else:
                st.write(f"- {k}: {v}")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 💡 Try These Questions:")
    st.markdown("- `mathematical formulation of LSTM`")
    st.markdown("- `How does the forget gate work?`")
    st.markdown("- `What is the vanishing gradient problem?`")
    st.markdown("- `LSTM vs RNN differences`")
    
    st.markdown("---")
    st.caption("Made with ❤️ using Streamlit and open-source AI.")
