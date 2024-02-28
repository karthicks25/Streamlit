import streamlit as st
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

# Load the model and tokenizer
model_name = "facebook/mbart-large-50-many-to-many-mmt"
model = MBartForConditionalGeneration.from_pretrained(model_name)
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)

def translate_text(input_text, source_lang, target_lang):
    # Preprocess the input with the target language code
    tokenizer.src_lang = source_lang
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate the translation
    tokenizer.tgt_lang = target_lang
    output = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[target_lang])

    # Decode the result and return
    return tokenizer.batch_decode(output, skip_special_tokens=True)[0]

st.title("Multilingual Translator")

# Textbox for input text
input_text = st.text_input("Text to Translate")

# Dropdown menus for languages
source_lang = st.selectbox("Source Language", ["English", "French", "German", "Spanish", "Hindi", "Tamil"])
target_lang = st.selectbox("Target Language", ["English", "French", "German", "Spanish", "Hindi", "Tamil"])
# Translate button
if st.button("Translate"):
    # Perform translation and display the result
    translated_text = translate_text(input_text, source_lang, target_lang)
    st.write("Translated Text:", translated_text)
