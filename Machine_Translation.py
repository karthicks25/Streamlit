import streamlit as st
from transformers import MarianMTModel, MarianTokenizer


def get_model(source_lang, target_lang):

  model_map = {
      ("en", "fr"): ("Helsinki-NLP/opus-mt-en-fr", MarianTokenizer),
      ("en", "ta"): ("Helsinki-NLP/opus-mt-en-ta", MarianTokenizer),  # Check availability
      ("en", "hi"): ("Helsinki-NLP/opus-mt-en-hi", MarianTokenizer),  # Check availability
      ("en", "de"): ("Helsinki-NLP/opus-mt-en-de", MarianTokenizer),
      ("fr", "en"): ("Helsinki-NLP/opus-mt-fr-en", MarianTokenizer),
      ("ta", "en"): ("Helsinki-NLP/opus-mt-ta-en", MarianTokenizer),  # Check availability
      ("hi", "en"): ("Helsinki-NLP/opus-mt-hi-en", MarianTokenizer),  # Check availability
      ("de", "en"): ("Helsinki-NLP/opus-mt-de-en", MarianTokenizer),
  }

  if (source_lang, target_lang) in model_map:
    model_name, tokenizer_cls = model_map[(source_lang, target_lang)]
    return tokenizer_cls.from_pretrained(model_name), MarianMTModel.from_pretrained(model_name)
  else:
    raise ValueError(f"Translation between {source_lang} and {target_lang} is not supported.")


def translate(text, source_lang, target_lang):
  """
  Translates text using the specified model.
  """
  tokenizer, model = get_model(source_lang, target_lang)
  inputs = tokenizer(text, return_tensors="pt", padding=True)
  translated = model.generate(**inputs)
  translation = tokenizer.decode(translated[0], skip_special_tokens=True)
  return translation


def main():
  st.title("Multilingual Translator")

  supported_languages = ["en", "fr", "ta", "hi", "de"]
    
  # Add dropdowns to select source and target languages
  source_lang = st.selectbox("Source Language", supported_languages)
  target_lang = st.selectbox("Target Language", supported_languages)

  text_to_translate = st.text_area("Enter text to translate:")

  if st.button("Translate"):
    if text_to_translate:
      try:
        translated_text = translate(text_to_translate, source_lang, target_lang)
        st.success("Translation:")
        st.write(translated_text)
      except ValueError as e:
        st.error(e)
    else:
      st.warning("Please enter some text to translate.")

if __name__ == "__main__":
  main()
