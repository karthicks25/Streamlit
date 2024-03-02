import streamlit as st
from transformers import MarianMTModel, MarianTokenizer

def get_model(source_lang, target_lang):

  model_map = {
      ("English", "French"): ("Helsinki-NLP/opus-mt-en-fr", MarianTokenizer),
      ("English", "German"): ("Helsinki-NLP/opus-mt-en-de", MarianTokenizer),
      ("French", "English"): ("Helsinki-NLP/opus-mt-fr-en", MarianTokenizer),
      ("German","English"): ("Helsinki-NLP/opus-mt-de-en", MarianTokenizer),
      ("German","French"): ("Helsinki-NLP/opus-mt-de-fr", MarianTokenizer),
      ("French","German"): ("Helsinki-NLP/opus-mt-fr-de", MarianTokenizer)
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

  supported_languages = ["English", "French", "German"]
    
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
