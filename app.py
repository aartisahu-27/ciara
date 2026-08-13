"""
CAIRA - Career Advancement Intelligent Resource Assistant
------------------------------------------------------------
A Streamlit chatbot that answers student questions about JAF/IAF
company data (CTC, branches, CGPA cutoff, sector, etc.) using the
Gemini API (free tier, for testing), grounded on your live
"JAF/IAF Extracted Data" Google Sheet.

SETUP (one-time):
1. Make sure your Google Sheet is shared as "Anyone with the link can view".
2. Fill in SPREADSHEET_ID, JAF_GID, IAF_GID below (see instructions under CONFIG).
3. Get a FREE Gemini API key from https://aistudio.google.com/apikey
   (no credit card required for the free tier).
4. Push this file + requirements.txt to a GitHub repo.
5. Deploy on https://share.streamlit.io, pointing at this repo/app.py.
6. In the Streamlit Cloud app's Settings -> Secrets, add:
       GEMINI_API_KEY = "AIza..."
7. Done -- you'll get a public URL like https://your-app.streamlit.app

NOTE ON DATA PRIVACY: Gemini's free tier may use prompts/responses to
improve Google's models. This is fine for a testing phase, but if you
later move to production with real student traffic and sensitive data,
consider switching to a paid tier (Gemini or Claude) where usage isn't
used for training by default. See the bottom of this file for how to
swap back to the Claude API later -- the rest of the app is unchanged.
"""

import streamlit as st
import pandas as pd
import google.generativeai as genai

# ==== CONFIG ====
# Find these in your Google Sheet's URL and each tab's URL:
#   https://docs.google.com/spreadsheets/d/SPREADSHEET_ID/edit#gid=SHEET_GID
# SPREADSHEET_ID is the long string after /d/
# SHEET_GID is the number after #gid= when you click each tab (JAF_Data, IAF_Data)
SPREADSHEET_ID = "11yUcfa66UgRDn4nYgSJz8F_ShSzakE_-ElRBR9htZr8"
JAF_GID = "329067337"                    # gid for the JAF_Data tab
IAF_GID = "208521765"  # gid for the IAF_Data tab
GEMINI_MODEL = "gemini-3.5-flash"  # current stable model as of Aug 2026; swap to "gemini-3.5-flash-lite" for higher free rate limits

def csv_url(sheet_id, gid):
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"

# ==== PAGE SETUP ====
st.set_page_config(page_title="CAIRA - Career Advancement Assistant", page_icon="🎓", layout="centered")
st.title("🎓 CAIRA")
st.caption("Career Advancement Intelligent Resource Assistant — ask me about JAF/IAF companies (CTC, branches, CGPA cutoff, and more).")

# ==== LOAD DATA (cached, refreshes every 10 minutes) ====
@st.cache_data(ttl=600, show_spinner="Loading latest JAF/IAF data...")
def load_data():
    jaf_df = pd.read_csv(csv_url(SPREADSHEET_ID, JAF_GID))
    jaf_df["Type"] = "JAF (Full-Time Placement)"
    iaf_df = pd.read_csv(csv_url(SPREADSHEET_ID, IAF_GID))
    iaf_df["Type"] = "IAF (Summer Internship)"
    combined = pd.concat([jaf_df, iaf_df], ignore_index=True)
    return combined

try:
    data = load_data()
except Exception as e:
    st.error(
        "Couldn't load the Google Sheet. Make sure SPREADSHEET_ID, JAF_GID, and "
        "IAF_GID are set correctly at the top of app.py, and that the sheet is "
        "shared as 'Anyone with the link can view'.\n\nDetails: " + str(e)
    )
    st.stop()

with st.sidebar:
    st.subheader("Data status")
    st.write(f"**{len(data)}** company entries loaded")
    st.write(f"JAF: {(data['Type'] == 'JAF (Full-Time Placement)').sum()} | "
             f"IAF: {(data['Type'] == 'IAF (Summer Internship)').sum()}")
    if st.button("🔄 Refresh data now"):
        st.cache_data.clear()
        st.rerun()
    st.caption("Data auto-refreshes every 10 minutes, or click above to force it.")

# ==== GEMINI CLIENT ====
if "GEMINI_API_KEY" not in st.secrets:
    st.error("GEMINI_API_KEY is not set in Streamlit secrets. Add it under App Settings -> Secrets.")
    st.stop()

genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

SYSTEM_PROMPT = """You are CAIRA, the Career Advancement placement assistant for BITS Pilani students.

You answer questions about company JAFs (Full-Time Placement job application forms) and
IAFs (Summer Internship application forms) -- covering CTC, eligible branches/courses,
CGPA cutoff, sector, eligibility criteria, gender/backlog policy, and related fields.

CRITICAL RULES:
- Answer ONLY using the data provided below. Do not guess, infer, or use outside knowledge about companies.
- Before answering, find the exact entry (or entries) matching the company/role the student asked about.
  Match company names loosely (ignore case, minor spelling differences) but never invent a match if
  none is close.
- If a company or field isn't in the data, say so clearly rather than making something up.
- If a company has multiple entries (e.g. multiple roles), list all of them or ask which role they mean.
- Quote CTC, CGPA cutoff, and other values EXACTLY as they appear in the data below -- do not round,
  reformat, or paraphrase numbers.
- Be concise. Use bullet points for multi-part answers.

Here is the current JAF/IAF data. Each company/role is a separate block below, with each field on its
own line in "Field: Value" format:

{data_blocks}
"""

def build_system_prompt(df: pd.DataFrame) -> str:
    # Format each row as a clearly delimited block (Field: Value per line) rather
    # than raw CSV -- this is far more reliable for the model to parse correctly,
    # especially with long, comma-containing text fields like Branches/Eligibility.
    blocks = []
    for _, row in df.iterrows():
        lines = [f"{col}: {row[col]}" for col in df.columns if pd.notna(row[col]) and str(row[col]).strip() != ""]
        blocks.append("---\n" + "\n".join(lines))
    return SYSTEM_PROMPT.format(data_blocks="\n\n".join(blocks))

# ==== CHAT UI ====
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about a company's CTC, branches, CGPA cutoff..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        try:
            model = genai.GenerativeModel(
                model_name=GEMINI_MODEL,
                system_instruction=build_system_prompt(data),
                generation_config=genai.types.GenerationConfig(temperature=0),
            )
            # Gemini uses "user"/"model" roles (not "assistant"), and history
            # excludes the message we're about to send.
            gemini_history = [
                {"role": ("model" if m["role"] == "assistant" else "user"), "parts": [m["content"]]}
                for m in st.session_state.messages[:-1]
            ]
            chat = model.start_chat(history=gemini_history)
            response = chat.send_message(prompt, stream=True)
            for chunk in response:
                if chunk.text:
                    full_response += chunk.text
                    placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"Something went wrong calling Gemini: {e}"
            placeholder.error(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

# ---------------------------------------------------------------------------
# TO SWITCH TO THE CLAUDE API LATER (e.g. once out of testing):
# 1. `pip install anthropic` and add it to requirements.txt
# 2. Replace the GEMINI CLIENT block above with:
#        import anthropic
#        client = anthropic.Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
# 3. Replace the try/except block above with a call to:
#        with client.messages.stream(
#            model="claude-sonnet-5", max_tokens=1000,
#            system=build_system_prompt(data),
#            messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
#        ) as stream:
#            for text in stream.text_stream:
#                full_response += text
#                placeholder.markdown(full_response + "▌")
# 4. Add ANTHROPIC_API_KEY to Streamlit secrets instead of GEMINI_API_KEY.
# ---------------------------------------------------------------------------
