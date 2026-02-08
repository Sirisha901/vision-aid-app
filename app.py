# app.py (Improved with proper working TTS in Streamlit)
import streamlit as st
from PIL import Image
import pytesseract
import spacy
import re
from gtts import gTTS
import io

# ---------- CONFIG ----------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
nlp = spacy.load("en_core_web_sm")

st.set_page_config(page_title="VisionAid", layout="centered")
st.title("🦯 VisionAid — An AI-Powered Assistant for visually impaired and illiterate")
st.write("Upload a printed document image. Below you will see (A) Cleaned full text and (B) Key information extracted.")

# ---------- HELPER FUNCTIONS ----------
def clean_ocr_text(raw: str) -> str:
    if not raw:
        return ""
    text = raw.replace('\r', '\n').replace('\t', ' ')
    text = re.sub(r'\n\s+\n', '\n\n', text)
    text = re.sub(r'[ \u00A0]+', ' ', text)
    subs = {'ﬁ':'fi','ﬂ':'fl','₹':'₹','R.s':'Rs','R S':'Rs','R S.':'Rs','rn':'m'}
    for a, b in subs.items():
        text = text.replace(a, b)
    text = re.sub(r'(?<=\d)O(?=\d)', '0', text)
    text = re.sub(r'(?<=\d)l(?=\d)', '1', text)
    text = re.sub(r'(?<=\d)I(?=\d)', '1', text)
    text = re.sub(r'(?<=\D)O(?=\d)', '0', text)
    text = re.sub(r'(?<=\d)O(?=\D)', '0', text)
    text = re.sub(r'\s+([,.:;%\)])', r'\1', text)
    text = re.sub(r'([(\[])\s+', r'\1', text)
    text = re.sub(r' {2,}', ' ', text)
    lines = [ln.strip() for ln in text.splitlines()]
    cleaned_lines = []
    for ln in lines:
        if ln == '':
            if cleaned_lines and cleaned_lines[-1] != '':
                cleaned_lines.append('')
        else:
            cleaned_lines.append(ln)
    return '\n'.join(cleaned_lines).strip()

def extract_key_info(cleaned_text: str) -> dict:
    data = {}
    text = cleaned_text.lower()
    doc = nlp(cleaned_text)
    ents = [(ent.text, ent.label_) for ent in doc.ents]
    dates_spacy = [t for t, lab in ents if lab == "DATE"]
    money_spacy = [t for t, lab in ents if lab == "MONEY"]

    # Dates
    date_patterns = [
        r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
        r'\b\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{2,4}\b',
        r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{1,2},?\s*\d{2,4}\b'
    ]
    dates = []
    for p in date_patterns:
        dates += [m.group(0) for m in re.finditer(p, text, flags=re.IGNORECASE)]
    dates += dates_spacy
    seen = set()
    dates = [x for x in dates if not (x in seen or seen.add(x))]
    if dates:
        data['Dates'] = dates

    # Amounts
    amount_patterns = [
        r'(?:rs\.?|inr|₹)\s?[0-9]+(?:[.,][0-9]{1,2})?(?:\s?/-)?',
        r'\b[0-9]{1,3}(?:,[0-9]{3})+(?:\.[0-9]{1,2})?\b',
        r'\b[0-9]+(?:\.[0-9]{1,2})?\s?(rupees|rs|inr)\b'
    ]
    amounts = []
    for p in amount_patterns:
        amounts += [m.group(0) for m in re.finditer(p, text, flags=re.IGNORECASE)]
    amounts += money_spacy
    amounts = [a.strip() for a in amounts]
    seen = set()
    amounts = [x for x in amounts if not (x in seen or seen.add(x))]
    if amounts:
        data['Amounts'] = amounts

    # Dosage / Instructions
    dosage_keywords = ['dosage','dose','take','tablet','capsule','mg','ml','twice','daily','once','after food','before food']
    dosage_candidates = []
    for ln in cleaned_text.splitlines():
        ln_lower = ln.lower()
        if any(k in ln_lower for k in dosage_keywords):
            dosage_candidates.append(ln.strip())
    seen = set()
    dosage_candidates = [x for x in dosage_candidates if not (x in seen or seen.add(x))]
    if dosage_candidates:
        data['Dosage/Instructions'] = dosage_candidates

    # Products/Meds
    product_candidates = [t for t, lab in ents if lab in ("PRODUCT","ORG","PERSON")]
    for ln in cleaned_text.splitlines():
        if any(word in ln.lower() for word in ['tablet','capsule','syrup','ointment','injection','mg']):
            product_candidates.append(ln.strip().split('-')[0].strip())
    seen = set()
    product_candidates = [x for x in product_candidates if not (x in seen or seen.add(x))]
    if product_candidates:
        data['Products/Meds'] = product_candidates

    # Keywords
    keywords_map = {
        'Expiry': ['expiry','exp','exp date','valid till','best before'],
        'MRP/Price': ['mrp','price','cost','amount','total','net payable','paid'],
        'Due': ['due date','payment due','last date']
    }
    for label, keys in keywords_map.items():
        for k in keys:
            if k in text:
                for ln in cleaned_text.splitlines():
                    if k in ln.lower():
                        data.setdefault(label, []).append(ln.strip())
                        break

    if not data:
        non_empty = [ln.strip() for ln in cleaned_text.splitlines() if ln.strip()]
        if non_empty:
            data['SummaryHint'] = non_empty[:3]

    return data

def build_natural_speech(found: dict, cleaned_text: str) -> str:
    if not found:
        snippet = cleaned_text.strip().replace('\n',' ')
        return snippet[:800] if len(snippet) > 0 else "No text to read."
    parts = []
    if 'Products/Meds' in found:
        parts.append("Medicines or items: "+ "; ".join(found['Products/Meds']))
    if 'Dosage/Instructions' in found:
        parts.append("Dosage/instructions: "+ "; ".join(found['Dosage/Instructions']))
    if 'Amounts' in found:
        parts.append("Amounts: "+ "; ".join(found['Amounts']))
    if 'Dates' in found:
        parts.append("Dates: "+ "; ".join(found['Dates']))
    for k, v in found.items():
        if k not in ('Products/Meds','Dosage/Instructions','Amounts','Dates'):
            parts.append(f"{k}: "+ "; ".join(v if isinstance(v, list) else [v]))
    return " . ".join(parts)

# ---------- UI ----------
uploaded_image = st.file_uploader("Upload an image (printed prescription, bill, label)", type=["png","jpg","jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    raw_text = pytesseract.image_to_string(image)
    cleaned = clean_ocr_text(raw_text)

    st.subheader("📄 Cleaned Document Text (readable):")
    st.text_area("Cleaned Full Text", cleaned, height=280)

    found = extract_key_info(cleaned)
    st.subheader("🔎 Key Information (auto-extracted):")
    if found:
        for k, v in found.items():
            if isinstance(v, list):
                for item in v:
                    st.markdown(f"- **{k}:** {item}")
            else:
                st.markdown(f"- **{k}:** {v}")
    else:
        st.info("No clear key fields found. See the cleaned full text above.")

    st.subheader("🔊 Listen (Text-to-Speech)")
    lang = st.selectbox("Language for speech:", ["English","Hindi","Telugu","Tamil","Odia"])
    lang_codes = {"English":"en","Hindi":"hi","Telugu":"te","Tamil":"ta","Odia":"or"}
    lang_code = lang_codes[lang]

    # Use st.form for reliable button-trigger
    with st.form("tts_form"):
        submit = st.form_submit_button("🎧 Generate & Play Voice")
        if submit:
            try:
                speech_text = build_natural_speech(found, cleaned)
                speech_text = speech_text[:5000]  # truncate for gTTS

                tts = gTTS(text=speech_text, lang=lang_code, slow=False)
                audio_buffer = io.BytesIO()
                tts.write_to_fp(audio_buffer)
                audio_buffer.seek(0)

                st.audio(audio_buffer, format="audio/mp3")
                st.success("Audio generated — listen above or download.")
                st.download_button("⬇️ Download Audio", data=audio_buffer, file_name="visistronaid_output.mp3")
            except Exception as e:
                st.error(f"Error generating audio: {e}")
else:
    st.info("Upload a printed prescription, bill, or label image to see cleaned text and key info.")
