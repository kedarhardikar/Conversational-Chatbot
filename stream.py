import os
import wave
import pyaudio
import streamlit as st
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_groq import ChatGroq
import pyttsx3
from gtts import gTTS
from playsound import playsound
import tempfile
from PyPDF2 import PdfReader
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

# --- Initialization ---
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")   #Convert text to embeddings
whisper_model = WhisperModel("base", compute_type="int8")  #Convert speech to text

chroma_client = chromadb.Client()  # In-memory ChromaDB instance, Stores document embeddings temporarily
collection = chroma_client.get_or_create_collection(name="my_document_embeddings")  #Creates or retrieves a collection

llm = ChatGroq(model="llama3-8b-8192", api_key=api_key) #Used to generate answers based on the context retrieved from vector DB.

engine = pyttsx3.init() #Initializes the pyttsx3 text-to-speech engine.
engine.setProperty('voice', engine.getProperty('voices')[0].id) #You could also pick a different voice (e.g., male/female, different accents) by changing the index


def speak(text, engine_type="gtts"):
    if engine_type == "gtts":   #Uses Google Text-to-Speech.
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            temp_path = fp.name
            tts.save(temp_path) #Creates a temporary .mp3 file to store the audio.

        playsound(temp_path)    #Plays the audio
        os.remove(temp_path)    #cleans up by deleting the file.


    else:   # Uses offline system voice.
        engine.say(text)    #Speaks the text using the offline voice engine (pyttsx3) initialized earlier
        engine.runAndWait()


# records voice input, saves it to a .wav file, and uses the Whisper ASR model to transcribe it to text.
def take_command_whisper():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 6
    FILENAME = "output.wav"

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = [stream.read(CHUNK) for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS))] #Reads data chunk by chunk and stores it into a list frames.


    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(FILENAME, 'wb') as wf:   #Saves the recorded frames into a standard WAV audio file.
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    segments, _ = whisper_model.transcribe(FILENAME)    #Transcribe the .wav file
    os.remove(FILENAME) #Remove the saved .wav file after transcription
    return " ".join([seg.text for seg in segments]).strip()




def chunk_text(text, chunk_size=200, overlap=40):
    sentences = sent_tokenize(text)
    chunks = []
    for i in range(0, len(sentences), chunk_size - overlap):
        chunk = " ".join(sentences[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

#it's taking uploaded PDFs, extracting text, chunking it, embedding each chunk, and adding it to your ChromaDB collection.
def embed_uploaded_documents(files):
    for file in files:
        if file.type == "application/pdf":
            reader = PdfReader(file)
            full_text = "\n".join([page.extract_text() or "" for page in reader.pages])
            
            chunks = chunk_text(full_text)
            for i, chunk in enumerate(chunks):
                if chunk.strip():
                    embedding = embedding_model.encode(chunk, convert_to_numpy=True).tolist()
                    collection.add(
                        documents=[chunk],
                        embeddings=[embedding],
                        ids=[f"doc_{file.name}_{i}"]
                    )
                            
def retrieve_relevant_chunks(query, top_k=5):
    query_embedding = embedding_model.encode(query, convert_to_numpy=True).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results["documents"][0] if results["documents"] else []


def generate_answer(query, retrieved_chunks, chat_history=[]):
    context = "\n\n".join(retrieved_chunks)
    history_text = "\n".join([f"User: {item['query']}\nAssistant: {item['response']}" for item in chat_history])
    prompt = f"""
You are an AI assistant that provides clear, concise, and informative answers based on provided context.

### Chat History:
{history_text}

### Context:
{context}

### Question:
{query}

### Instructions:
- Summarize the relevant information from the context to answer the question.
- Provide a clear, structured, and fact-based response.
- Answer in 2-3 lines if possible
- Answer in brief only if necessary
- If the context does not contain enough information, say \"The provided context does not contain sufficient details.\"

### Response:
"""
    response = llm.invoke(prompt)
    return response.content

# --- Streamlit App ---
st.set_page_config(page_title="🎤 Voice Assistant", layout="centered")
st.title("🎤 Voice Assistant")
st.write("Upload documents to query from and ask questions via voice or text.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "tts_engine" not in st.session_state:
    st.session_state.tts_engine = "gtts"

uploaded_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    embed_uploaded_documents(uploaded_files)
    st.success("Documents embedded into vector store.")

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("🎙 Speak"):
        st.write("Listening...")
        user_query = take_command_whisper()
        st.write(f"You said: {user_query}")
        speak(f"Did you say: {user_query}? Say yes or no.", st.session_state.tts_engine)

        # Loop until the speech is confirmed
        while True:
            confirm = take_command_whisper().lower()
            if "no" in confirm:
                speak("Okay, please try again.", st.session_state.tts_engine)
                st.warning("Speech not confirmed. Try again.")
                user_query = take_command_whisper()  # Re-listen if user says "no"
                st.write(f"You said: {user_query}")
                speak(f"Did you say: {user_query}? Say yes or no.", st.session_state.tts_engine)
            else:
                # When "yes" is confirmed, proceed with answering
                retrieved_chunks = retrieve_relevant_chunks(user_query)
                response = generate_answer(user_query, retrieved_chunks, st.session_state.chat_history)
                st.session_state.chat_history.append({"query": user_query, "response": response})
                st.markdown(f"<b>Assistant:</b> {response}", unsafe_allow_html=True)
                speak(response, st.session_state.tts_engine)
                break  # Exit the loop once the user confirms


with col2:
    if st.button("💬 Continue Conversation"):
        st.info("Say your follow-up now...")
        user_query = take_command_whisper()
        st.write(f"You said: {user_query}")
        retrieved_chunks = retrieve_relevant_chunks(user_query)
        response = generate_answer(user_query, retrieved_chunks, st.session_state.chat_history)
        st.session_state.chat_history.append({"query": user_query, "response": response})
        st.markdown(f"<b>Assistant:</b> {response}", unsafe_allow_html=True)
        speak(response, st.session_state.tts_engine)

with col3:
    if st.button("❌ End Conversation"):
        st.session_state.chat_history = []
        st.success("Conversation history cleared.")
        speak("Conversation ended. Thank you!", st.session_state.tts_engine)

st.markdown("---")
st.subheader("🧠 Chat History")
for chat in st.session_state.chat_history:
    st.markdown(f"<b>You:</b> {chat['query']}")
    st.markdown(f"<b>Assistant:</b> {chat['response']}")

st.markdown("---")
with st.expander("⚙️ Settings"):
    tts_option = st.selectbox("Choose TTS Engine", ["gtts", "pyttsx3"])
    st.session_state.tts_engine = tts_option
    st.success(f"TTS Engine set to: {tts_option}")

    with st.form("TextInput"):
        text_input = st.text_input("Or ask your question by typing:")
        submitted = st.form_submit_button("Ask")
        if submitted and text_input:
            chunks = retrieve_relevant_chunks(text_input)
            response = generate_answer(text_input, chunks, st.session_state.chat_history)
            st.session_state.chat_history.append({"query": text_input, "response": response})
            st.markdown(f"<b>Assistant:</b> {response}", unsafe_allow_html=True)
            speak(response, st.session_state.tts_engine)
