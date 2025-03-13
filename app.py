import streamlit as st
import base64
import os
from gtts import gTTS
from PIL import Image
import cv2
import pytesseract
import easyocr
import threading
import httpx
import google.generativeai as genai
from dotenv import load_dotenv

# --- Configuration of Gemini API ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINIKEY"))
model = genai.GenerativeModel(model_name="gemini-1.5-pro")

# --- Supported languages for TTS ---
LANGUAGES = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Dutch": "nl",
    "Russian": "ru",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese (Simplified)": "zh-CN",
    "Hindi": "hi",
}

# --- Page Configurations ---
st.set_page_config(page_title="LexiRead", layout="wide")

# --- Initialize font in session state ---
if "font" not in st.session_state:
    st.session_state.font = "Arial"


# --- Function to toggle font ---
def toggle_font():
    st.session_state.font = (
        "OpenDyslexic" if st.session_state.font == "Arial" else "Arial"
    )


# --- Font Toggle Button ---
if st.button(f"⏳"):
    toggle_font()

# --- Apply Font Styling ---
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=OpenDyslexic&display=swap');
    
    body, .stApp, .stTextInput, .stTextArea, .stButton, .stRadio, .stSelectbox {{
        font-family: {st.session_state.font}, sans-serif !important;
    }}
    </style>
""",
    unsafe_allow_html=True,
)

# Custom CSS for Styling with Animations
st.markdown(
    f"""
    <style>
    .stApp {{
        max-width: 1200px;
        margin: 0 auto;
        background-color: #EDE7F6;
        color: #4A148C;
        font-family: '{st.session_state.font}', Arial, sans-serif;
        animation: fadeIn 1s ease-in-out;
    }}
    @keyframes fadeIn {{
        from {{ opacity: 0; }}
        to {{ opacity: 1; }}
    }}
    h1 {{
        font-size: 4rem;
        color: #1E88E5;
        margin-bottom: 0.5rem;
        text-align: center;
    }}
    h2, h3, p, label, .stButton button {{
        font-family: '{st.session_state.font}', Arial, sans-serif !important;
        text-align: center;
    }}
    textarea, .stButton button {{
        font-family: '{st.session_state.font}', Arial, sans-serif !important;
        text-align: center;
    }}
    </style>
""",
    unsafe_allow_html=True,
)


# --- Load OpenDyslexic Font from Local File ---
FONT_PATH = "OpenDyslexic-Regular.otf"


def get_font_base64(font_path):
    with open(font_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


if os.path.exists(FONT_PATH):
    FONT_BASE64 = get_font_base64(FONT_PATH)

    st.markdown(
        f"""
        <style>
        @font-face {{
            font-family: 'OpenDyslexic';
            src: url(data:font/opentype;charset=utf-8;base64,{FONT_BASE64}) format('opentype');
        }}
        </style>
    """,
        unsafe_allow_html=True,
    )
else:
    st.error(
        f"⚠ OpenDyslexic-Regular.otf not found at {FONT_PATH}! Make sure it's in the correct folder."
    )

# --- Title Section ---
st.markdown("<h1>📖 LexiRead</h1>", unsafe_allow_html=True)
st.markdown(
    "<h2>Read Easy, Read your Way!</h2><hr style='height:6px;'>", unsafe_allow_html=True
)

# --- Navigation Bar ---
section = st.radio(
    "",
    ("Home", "Features", "Contact Us", "Know More"),
    horizontal=True,
    label_visibility="collapsed",
)

# --- Home Section ---
if section == "Home":
    st.markdown("<h3>About Us</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        <p>At LexiRead, we’re here to support people with dyslexia.</p>
        <p>We know that dyslexia can make certain tasks harder, but we believe everyone deserves the tools and support to succeed.</p>
        <p>Our platform offers easy-to-use resources, tips, and a community that helps you manage challenges and unlock your full potential.</p>
        <p>Whether you're navigating school, work, or everyday life, we’re here to help make things easier and show you that thinking differently is a real strength.</p>
        <p>Together, we can turn challenges into opportunities.</p>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("<h3>🌟 Features</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        <p>✔ Dyslexia-Friendly Fonts for better readability.</p>
        <p>✔ Customize your background colors for a more comfortable experience.</p>
        <p>✔ Adjust text spacing to reduce visual crowding.</p>
        <p>✔ Use Text-to-Speech for an audio version of the text, perfect for auditory learners!</p>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("<hr>", unsafe_allow_html=True)

# --- Features Section ---
if section == "Features":
    feature_option = st.selectbox(
        "Select a Feature",
        ("Extract Text from Image", "Text to Speech", "Dyslexic-Friendly Text"),
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    if feature_option == "Extract Text from Image":
        st.markdown(
            "<div style='background-color:#BBDEFB; padding:10px;'>",
            unsafe_allow_html=True,
        )

        uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            st.image(image, use_container_width=True)

        # --- Extraction of text using Gemini API ---
        def get_gemini(image_path):
            image = Image.open(image_path)
            prompt = f"""you are supposed to extract the text from the uploaded image, 
            ignore the background details, read all types of fonts, calligraphy and fancy texts, 
            if there are any errors or missing words fix it along with grammar"""
            response = model.generate_content([prompt, image])
            return response.text

        # --- Extraction of text using Easyocr ---
        def extract_handwritten_text_from_image(image_path):
            # Read the image using OpenCV
            image = cv2.imread(image_path)

            # Convert the image to grayscale (helpful for OCR accuracy)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Optional: Apply some noise reduction or thresholding if needed
            _, thresh_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)

            # Use EasyOCR to extract text
            reader = easyocr.Reader(["en"])
            result = reader.readtext(
                thresh_image, detail=0
            )  # detail=0 returns only the text
            return " ".join(result)

        if uploaded_image is not None:
            with open("temp_image.jpg", "wb") as f:
                f.write(uploaded_image.getbuffer())

            try:
                extracted_text_easyocr = extract_handwritten_text_from_image(
                    "temp_image.jpg"
                )
                extracted_text_gemini = get_gemini("temp_image.jpg")
                final_text = (
                    extracted_text_easyocr
                    if len(extracted_text_easyocr) > len(extracted_text_gemini)
                    else extracted_text_gemini
                )

                st.subheader("Extracted Text:")
                st.markdown(
                    f'<div style="font-family: OpenDyslexic;">{final_text}</div>',
                    unsafe_allow_html=True,
                )

                selected_lang_user = st.selectbox(
                    "Select Language for Speech",
                    options=list(LANGUAGES.keys()),
                    key="lang_user",
                )
                if st.button("🔊 Convert to Speech"):
                    if final_text.strip():
                        try:
                            # --- Converting Text to Speech and in Specified Foriegn Language ---
                            tts = gTTS(
                                text=final_text,
                                lang=LANGUAGES[selected_lang_user],
                                slow=False,
                            )
                            tts.save("response.mp3")
                            with open("response.mp3", "rb") as audio_file:
                                st.audio(audio_file.read(), format="audio/mp3")
                        except Exception as e:
                            st.error(
                                f"An error occurred during text-to-speech conversion: {e}"
                            )
                    else:
                        st.warning("Please enter some text to convert to speech.")

            except Exception as e:
                st.error(f"An error occurred while extracting text: {e}")

        st.markdown("</div>", unsafe_allow_html=True)

    # --- Conversion of uploaded text to dyslexia friendly font ---
    elif feature_option == "Text to Speech":
        st.markdown(
            "<div style='background-color:#E3F2FD; padding:10px;'>",
            unsafe_allow_html=True,
        )
        user_text = st.text_area("Enter text below:", height=150)
        # st.markdown("<h3 class='dyslexia-text'>Enter Text Below:</h3>", unsafe_allow_html=True)
        selected_lang_user = st.selectbox(
            "Select Language for Speech",
            options=list(LANGUAGES.keys()),
            key="lang_user",
        )
        if st.button("🔊 Convert to Speech"):
            if user_text.strip():
                try:
                    tts = gTTS(
                        text=user_text, lang=LANGUAGES[selected_lang_user], slow=False
                    )
                    tts.save("response.mp3")
                    with open("response.mp3", "rb") as audio_file:
                        st.audio(audio_file.read(), format="audio/mp3")
                except Exception as e:
                    st.error(f"An error occurred during text-to-speech conversion: {e}")
            else:
                st.warning("Please enter some text to convert to speech.")

    elif feature_option == "Dyslexic-Friendly Text":
        st.markdown(
            "<div style='background-color:#C8E6C9; padding:10px;'>",
            unsafe_allow_html=True,
        )
        dyslexic_text = st.text_area("Enter text to format:", height=150)
        if st.button("Format Text"):
            if dyslexic_text:
                st.markdown(
                    f'<div style="font-family: OpenDyslexic;">{dyslexic_text}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.warning("Please enter some text first!")
        st.markdown("</div>", unsafe_allow_html=True)

# --- Know More Section ---
elif section == "Know More":
    st.markdown("<h3>📚 Learn More About Dyslexia</h3>", unsafe_allow_html=True)

    # Questions and Answers
    questions_and_answers = [
        (
            "What is Dyslexia?",
            "Dyslexia is a specific learning disability that affects reading, spelling, and writing. It is characterized by difficulty in recognizing and decoding words.",
        ),
        (
            "What causes Dyslexia?",
            "Dyslexia is believed to be caused by genetic and neurobiological factors. It affects how the brain processes written and spoken language.",
        ),
        (
            "Can Dyslexia be treated?",
            "While there is no cure for dyslexia, individuals can receive specialized interventions and accommodations that help them manage the condition and succeed academically.",
        ),
        (
            "Is Dyslexia common?",
            "Yes, dyslexia affects approximately 1 in 10 people. It is one of the most common learning disabilities, and early diagnosis and intervention can lead to better outcomes.",
        ),
    ]

    for question, answer in questions_and_answers:
        st.markdown(
            f"<div class='question' style='font-family: OpenDyslexic;'>{question}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='answer' style='font-family: OpenDyslexic;'>{answer}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

# --- Contact Us Section ---
elif section == "Contact Us":
    st.markdown("<h3>📩 Contact Us</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        <p>If you have any questions, concerns, or feedback, feel free to reach out to us!</p>
        <div class="contact-block">
            <p><strong>Smriti Sreeram:</strong> smriti.sreeram2024@vitstudent.ac.in</p>
            <p><strong>Parneeca Mahale:</strong> parneeca.prasad2024@vitstudent.ac.in</p>
            <p><strong>Vansh Sharma:</strong> vansh.sharma2024@vitstudent.ac.in</p>
            <p><strong>Sanidhya Labh:</strong> sanidhya.labh2024@vitstudent.ac.in</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

# --- Footer ---
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center; font-family: OpenDyslexic;'>Disability? Nah, this ability💪🏼</p>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center; font-family: OpenDyslexic;'>© LexiRead | Empowering Reading for Everyone</p>",
    unsafe_allow_html=True,
)
