import streamlit as st
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load the model and tokenizer (consider a smaller model if possible)
model_name = "alirezamsh/small100"  # Smaller model suggestion
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def translate_text(input_text, source_lang, target_lang):
    # Preprocess the input with the target language code
    inputs = tokenizer(f"<2{target_lang}> {input_text}", return_tensors="pt")

    # Generate the translation
    output = model.generate(**inputs)

    # Decode the result and return
    return tokenizer.batch_decode(output, skip_special_tokens=True)[0]

st.title("Multilingual Translator")

# Textbox for input text
input_text = st.text_input("Text to Translate")

# Dropdown menus for languages
source_lang = st.selectbox("Source Language", ["en", "fr", "de", "es", "hi", "ta"])  # Use listed language codes
target_lang = st.selectbox("Target Language", ["en", "fr", "de", "es", "hi", "ta"])  # Use listed language codes

# Translate button
if st.button("Translate"):
    # Perform translation and display the result
    translated_text = translate_text(input_text, source_lang, target_lang)
    st.write("Translated Text:", translated_text)

# Optionally experiment with caching (carefully consider potential drawbacks)
@st.cache  # Uncomment to try caching
def translate_text_cached(input_text, source_lang, target_lang):
    """Cached translation function (optional)."""
    return translate_text(input_text, source_lang, target_lang)

