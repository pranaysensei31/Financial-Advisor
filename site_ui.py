import streamlit as st

def apply_site_theme():
    st.markdown("""
    <style>
      /* --- Hide Streamlit UI --- */
      #MainMenu, header, footer {visibility: hidden;}
      section[data-testid="stSidebar"]{display:none;}
      div[data-testid="stToolbar"]{display:none !important;}

      /* --- Page background --- */
      .stApp{
        background: #070A0F !important;
        color: #E5E7EB !important;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      }

      /* --- Main container --- */
      .block-container{
        max-width: 1260px;
        padding-top: 1.2rem;
        padding-bottom: 2.5rem;
      }

      /* --- Typography --- */
      h1,h2,h3,h4 { color:#F9FAFB !important; font-weight: 950 !important; letter-spacing:-0.4px; }
      p,li,span,div { color:#D1D5DB; }

      /* --- Inputs --- */
      div[data-baseweb="input"] > div{
        background:#0B1220 !important;
        border:1px solid rgba(255,255,255,0.08) !important;
        border-radius: 14px !important;
      }
      div[data-baseweb="input"] input{
        color:#F9FAFB !important;
        font-weight:900 !important;
      }

      /* --- Buttons --- */
      .stButton > button{
        border-radius: 14px !important;
        height: 44px !important;
        font-weight: 950 !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        background: #0B1220 !important;
        color:#F9FAFB !important;
      }
      .stButton > button:hover{
        border-color: rgba(59,130,246,0.45) !important;
        box-shadow: 0 12px 30px rgba(59,130,246,0.12) !important;
        transform: translateY(-1px);
        transition: 0.18s;
      }

      /* --- Metrics --- */
      div[data-testid="stMetricValue"] *{ color:#F9FAFB !important; font-weight: 1000 !important; }
      div[data-testid="stMetricLabel"] *{ color:#94A3B8 !important; font-weight: 900 !important; }
    </style>
    """, unsafe_allow_html=True)
