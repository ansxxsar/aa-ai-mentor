import streamlit as st
from main import agent_executor, parser, llm
from langchain_core.messages import HumanMessage, AIMessage
from PIL import Image
import base64
from io import BytesIO


def encode_image(image):
    buffered = BytesIO()
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


st.set_page_config(page_title="Ansar's AI Mentor", page_icon="🎓", layout="wide")

# ---- Session State Init ----
if "all_chats" not in st.session_state:
    st.session_state.all_chats = {"Chat 1": []}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "Chat 1"

# ---- Sidebar ----
with st.sidebar:
    st.header("⚙️ Mentor Settings")
    temperature = st.slider("Creativity (Temperature)", 0.0, 1.0, 0.2)
    st.info("Low temperature = factual research. High temperature = brainstorming.")

    st.divider()

    # ---- Chat Sessions ----
    st.subheader("💬 Chat Sessions")

    if st.button("➕ New Chat"):
        new_chat_name = f"Chat {len(st.session_state.all_chats) + 1}"
        st.session_state.all_chats[new_chat_name] = []
        st.session_state.current_chat = new_chat_name
        st.rerun()

    for chat_name in list(st.session_state.all_chats.keys()):
        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(
                f"{'▶ ' if chat_name == st.session_state.current_chat else ''}{chat_name}",
                key=f"select_{chat_name}",
                use_container_width=True
            ):
                st.session_state.current_chat = chat_name
                st.rerun()
        with col2:
            if st.button("🗑", key=f"delete_{chat_name}"):
                del st.session_state.all_chats[chat_name]
                if st.session_state.current_chat == chat_name:
                    if st.session_state.all_chats:
                        st.session_state.current_chat = list(st.session_state.all_chats.keys())[0]
                    else:
                        st.session_state.all_chats = {"Chat 1": []}
                        st.session_state.current_chat = "Chat 1"
                st.rerun()

    st.divider()

    # ---- Export Chat ----
    st.subheader("📄 Export Chat")
    if st.button("⬇️ Download as PDF"):
        from export import export_chat_to_pdf
        current_messages = st.session_state.all_chats[st.session_state.current_chat]
        if len(current_messages) == 0:
            st.warning("No messages to export yet!")
        else:
            pdf_bytes = export_chat_to_pdf(
                st.session_state.current_chat,
                current_messages
            )
            st.download_button(
                label="📥 Click to Save PDF",
                data=pdf_bytes,
                file_name=f"{st.session_state.current_chat}.pdf",
                mime="application/pdf"
            )

    st.divider()

    # ---- Course Materials RAG ----
    st.subheader("📚 Course Materials")
    course_file = st.file_uploader(
        "Upload lecture notes or course PDFs",
        type=["pdf"],
        key="course_pdf"
    )
    if course_file:
        from rag import process_pdf
        with st.spinner("Processing PDF..."):
            num_chunks = process_pdf(course_file)
            st.success(f"✅ Processed {num_chunks} chunks from your PDF!")

    st.divider()

    # ---- Image Upload ----
    st.subheader("📸 Upload Materials")
    uploaded_file = st.file_uploader(
        "Code screenshot or system diagram",
        type=["png", "jpg", "jpeg"],
        key="image_upload"
    )
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded for analysis")


# ---- Main Area ----
st.title("🎓 AA AI Mentor")
st.markdown(f"**Current chat:** {st.session_state.current_chat}")
st.markdown("""
An intelligent learning support system for the **Informatics and Digital Pedagogy** Master's program.  
This agent can analyze text, search academic papers, see screenshots, and search your course materials.
""")

# ---- Image Analysis ----
if uploaded_file and st.button("🔍 Analyze Image"):
    with st.chat_message("assistant"):
        with st.spinner("Analyzing the image..."):
            image = Image.open(uploaded_file)
            base64_img = encode_image(image)

            vision_response = llm.invoke([
                {"role": "system", "content": "You are a technical mentor. Analyze the image (code, diagram, or error) and provide clear academic and technical recommendations."},
                {"role": "user", "content": [
                    {"type": "text", "text": "What is wrong in this image/screenshot?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                ]}
            ])
            st.markdown("### 🔍 Visual Analysis Result:")
            st.write(vision_response.content)

# ---- Display Current Chat Messages ----
messages = st.session_state.all_chats[st.session_state.current_chat]

for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---- Chat Input ----
if prompt := st.chat_input("e.g. 'What does my lecture say about RAG?' or 'Schedule Thesis Review tomorrow at 10AM'"):

    messages.append({"role": "user", "content": prompt})

    # Auto rename chat based on first message
    if len(messages) == 1:
        chat_title = prompt[:25] + "..." if len(prompt) > 25 else prompt
        old_name = st.session_state.current_chat
        st.session_state.all_chats[chat_title] = st.session_state.all_chats.pop(old_name)
        st.session_state.current_chat = chat_title

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Researching and analyzing...", expanded=True) as status:
            try:
                st.write("Accessing tools (ArXiv/Tavily/Calendar/Course Materials)...")

                llm.temperature = temperature

                # Build chat history
                chat_history = []
                for msg in messages[:-1]:
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        chat_history.append(AIMessage(content=msg["content"]))

                raw_response = agent_executor.invoke({
                    "query": prompt,
                    "chat_history": chat_history
                })

                output = raw_response.get("output")

                if isinstance(output, str):
                    try:
                        data = parser.parse(output)
                        is_structured = True
                    except:
                        is_structured = False
                else:
                    data = output
                    is_structured = True

                status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

                if is_structured:
                    st.subheader("📊 Mentor Analysis")
                    st.write(data.analysis)

                    with st.expander("🛠 Technical Documentation"):
                        st.markdown(data.documentation_snippet)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("💡 Suggested Improvements")
                        for improvement in data.suggested_improvements:
                            st.markdown(f"- {improvement}")
                    with col2:
                        st.subheader("📚 Academic References (ArXiv)")
                        for ref in data.academic_references:
                            st.markdown(f"- {ref}")

                    messages.append({"role": "assistant", "content": data.analysis})

                else:
                    if "http" in str(output):
                        import re
                        urls = re.findall(r'https?://\S+', str(output))
                        st.write(output)
                        for url in urls:
                            st.link_button("📅 Open in Google Calendar", url)
                    else:
                        st.write(output)

                    messages.append({"role": "assistant", "content": str(output)})

            except Exception as e:
                status.update(label="❌ An error occurred", state="error")
                st.error(f"Error: {e}")
                if 'raw_response' in locals():
                    st.write("Raw data:", raw_response.get("output"))