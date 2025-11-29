"""
QARM II - Portfolio Optimizer
Landing Page
"""

import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from utils.helpers import load_css, init_session_state


st.set_page_config(
    page_title="QARM II - Portfolio Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

load_css()
init_session_state()

st.markdown("""
<style>
    /* Hide sidebar completely on landing page */
    [data-testid="stSidebar"] {
        display: none;
    }

    /* Hide sidebar toggle button */
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    /* Remove all top spacing from Streamlit */
    .main .block-container {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    /* Target Streamlit's app view container */
    .stApp > header {
        display: none !important;
    }

    .stApp [data-testid="stAppViewContainer"] {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    .stApp [data-testid="stAppViewBlockContainer"] {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    /* Remove default Streamlit top padding */
    .stMainBlockContainer {
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    section[data-testid="stMain"] {
        padding-top: 0 !important;
    }

    .stVerticalBlock {
        gap: 0 !important;
    }

    /* Remove spacing around hr separator */
    .stMarkdown:has(hr) {
        margin: 0 !important;
        padding: 0 !important;
    }

    .element-container:has(hr) {
        margin: 0 !important;
        padding: 0 !important;
    }

    hr {
        margin: 0.25rem 0 !important;
        padding: 0 !important;
    }

    /* Move content up with negative margin if needed */
    .landing-container {
        margin-top: 0.5rem !important;
    }

    header[data-testid="stHeader"] {
        display: none !important;
    }

    /* Landing page container */
    .landing-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 0 2rem 1rem 2rem;
    }

    /* Header section */
    .landing-header {
        text-align: center;
        padding: 0 !important;
        margin: 0 !important;
    }

    .landing-title {
        font-size: clamp(1.8rem, 2.5vw, 2.5rem) !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1.1 !important;
    }

    .landing-subtitle {
        font-size: clamp(0.95rem, 1.1vw, 1.1rem) !important;
        color: var(--text-secondary) !important;
        margin: 0 !important;
        padding: 0 0 0.1rem 0 !important;
        line-height: 1.2 !important;
    }

    /* Override Streamlit's default markdown spacing */
    .landing-header h1 {
        margin: 0 !important;
        padding: 0 !important;
    }

    .landing-header p {
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Header row with logos */
    .header-row {
        width: 100%;
        max-width: 1400px;
        margin: 0 auto;
        display: flex !important;
        align-items: flex-start !important;
    }

    .header-row > [data-testid="column"] {
        display: flex !important;
        flex: 1 !important;
    }

    .logo-left {
        display: flex !important;
        align-items: flex-start !important;
        justify-content: flex-start !important;
        padding-top: 1.5rem !important;
        padding-left: 0 !important;
        width: 100% !important;
    }

    .logo-left > div[data-testid="stImage"] {
        margin-left: 0 !important;
        width: 120px !important;
        max-width: 120px !important;
    }

    .header-center {
        display: flex !important;
        align-items: flex-start !important;
        justify-content: center !important;
        width: 100% !important;
    }

    .logo-right {
        display: flex !important;
        align-items: flex-start !important;
        justify-content: flex-end !important;
        padding-top: 1.5rem !important;
        padding-right: 0 !important;
        width: 100% !important;
    }

    .logo-right > div[data-testid="stImage"] {
        margin-right: 0 !important;
        margin-left: auto !important;
        width: 120px !important;
        max-width: 120px !important;
    }

    .logo-right img {
        margin-right: 0 !important;
        display: block !important;
    }

    /* Flip card container */
    .flip-cards-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 3rem;
        margin: 1rem 0;
        flex-wrap: wrap;
        max-width: 1400px;
        margin-left: auto;
        margin-right: auto;
    }

    /* Flip card */
    .flip-card {
        background-color: transparent;
        width: 100%;
        max-width: 607px;
        height: 405px;
        perspective: 1000px;
        cursor: pointer;
        flex: 0 1 auto;
    }

    .flip-card-inner {
        position: relative;
        width: 100%;
        height: 100%;
        text-align: center;
        transition: transform 0.6s;
        transform-style: preserve-3d;
    }

    .flip-card:hover .flip-card-inner {
        transform: rotateY(180deg);
    }

    .flip-card-front, .flip-card-back {
        position: absolute;
        width: 100%;
        height: 100%;
        -webkit-backface-visibility: hidden;
        backface-visibility: hidden;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }

    .flip-card-front {
        background-color: var(--bg-elevated);
        overflow: hidden;
    }

    .flip-card-front img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        border-radius: 12px;
    }

    .flip-card-back {
        background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-hover) 100%);
        transform: rotateY(180deg);
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 2rem;
    }

    .flip-card-back-content {
        max-width: 90%;
    }

    .flip-card-back h3 {
        font-size: 1.75rem;
        margin-bottom: 1rem;
        font-weight: 600;
        color: #000000 !important;
    }

    .flip-card-back p {
        font-size: 1.1rem;
        line-height: 1.6;
        color: #000000 !important;
    }

    /* CTA Button */
    .cta-container {
        text-align: center;
        margin: 1.5rem 0 1rem 0;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .cta-container > div {
        display: flex;
        justify-content: center;
    }

    .cta-button {
        display: inline-block;
        padding: 1rem 3rem;
        font-size: 1.25rem;
        font-weight: 600;
        color: white !important;
        background: linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-hover) 100%);
        border: none;
        border-radius: 8px;
        text-decoration: none !important;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(25, 118, 210, 0.3);
    }

    .cta-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(25, 118, 210, 0.4);
    }

    /* Footer */
    .landing-footer {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
        margin-top: 1rem;
        border-top: 1px solid var(--border-primary);
        color: var(--text-muted);
    }

    /* Responsive adjustments for very large screens */
    @media (min-width: 1600px) {
        .flip-cards-container {
            gap: 2rem;
            max-width: 1300px;
        }

        .flip-card {
            max-width: 550px;
            height: 367px;
        }

        .landing-container {
            max-width: 1300px;
        }
    }

    /* Responsive adjustments for medium screens */
    @media (max-width: 1200px) {
        .flip-card {
            max-width: 500px;
            height: 334px;
        }

        .flip-cards-container {
            gap: 2rem;
        }
    }

    /* Responsive adjustments for tablets */
    @media (max-width: 768px) {
        .header-row {
            flex-direction: column;
            align-items: center;
        }

        .logo-left, .logo-right {
            display: none !important;
        }

        .header-center {
            padding: 1rem 0;
        }

        .flip-card {
            max-width: 100%;
            height: auto;
            aspect-ratio: 3/2;
        }
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Landing page"""

    import base64

    app_dir = Path(__file__).parent
    qarm_logo_path = app_dir / "imports" / "QARMII_Logo.png"
    unil_logo_path = app_dir / "imports" / "UNIL_Logo.png"
    bl_img_path = app_dir / "imports" / "BlackLitterman_Img.png"
    nlp_img_path = app_dir / "imports" / "NLP_Img.png"

    def get_base64_image(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    qarm_logo_base64 = get_base64_image(qarm_logo_path)
    unil_logo_base64 = get_base64_image(unil_logo_path)

    st.markdown('<div class="landing-container">', unsafe_allow_html=True)

    # Header with logos
    st.markdown(f"""
    <div class="header-row" style="display: flex; align-items: flex-start; justify-content: space-between; width: 100%; padding-top: 0; margin-top: 0;">
        <div class="logo-left" style="flex: 0 0 120px;">
            <img src="data:image/png;base64,{qarm_logo_base64}" alt="QARM II Logo" style="width: 120px; display: block;">
        </div>
        <div class="header-center" style="flex: 1; text-align: center;">
            <div class="landing-header" style="padding: 0; margin: 0;">
                <h1 class="landing-title" style="font-size: 2.5rem; font-weight: 700; margin: 0; padding: 0; line-height: 1.1;">QARM II Portfolio Optimizer</h1>
                <p class="landing-subtitle" style="font-size: 1.1rem; margin: 0; padding: 0; line-height: 1.2;">Advanced portfolio optimization combining Black-Litterman methodology with NLP-driven sentiment analysis</p>
            </div>
        </div>
        <div class="logo-right" style="flex: 0 0 120px; display: flex; justify-content: flex-end;">
            <img src="data:image/png;base64,{unil_logo_base64}" alt="UNIL Logo" style="width: 120px; display: block;">
        </div>
    </div>
    <hr style="border: none; border-top: 1px solid var(--border-primary); margin: 1rem 0 0; padding: 0;">
    """, unsafe_allow_html=True)

    # Flip Cards Section
    bl_img_base64 = get_base64_image(bl_img_path)
    nlp_img_base64 = get_base64_image(nlp_img_path)

    st.markdown('<div class="flip-cards-container">', unsafe_allow_html=True)

    card_col1, card_col2 = st.columns(2, gap="large")

    with card_col1:
        st.markdown(f"""
        <div class="flip-card">
            <div class="flip-card-inner">
                <div class="flip-card-front">
                    <img src="data:image/png;base64,{bl_img_base64}" alt="Black-Litterman Model">
                </div>
                <div class="flip-card-back">
                    <div class="flip-card-back-content">
                        <h3>Black-Litterman Model</h3>
                        <p>The model derives prior expected returns from global market-cap weights (implied equilibrium returns).</p>
                        <p>Your custom views, scaled by confidence, are integrated through a Bayesian update to generate posterior returns. These flow into a mean-variance optimizer to produce stable, well-regularized portfolio weights with reduced estimation error and improved risk allocation.</p>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with card_col2:
        st.markdown(f"""
        <div class="flip-card">
            <div class="flip-card-inner">
                <div class="flip-card-front">
                    <img src="data:image/png;base64,{nlp_img_base64}" alt="NLP Sentiment Analysis">
                </div>
                <div class="flip-card-back">
                    <div class="flip-card-back-content">
                        <h3>NLP Sentiment Analysis</h3>
                        <p>Claude Haiku (Anthropic's LLM) processes financial news headlines to classify sentiment as bullish, neutral, or bearish with a confidence score.</p>
                        <p>Scores are aggregated into a normalized sentiment metric [-1,1] for each asset. This signal is mapped into a view vector with confidence, providing systematic, data-driven adjustments to the Black-Litterman posterior returns.</p>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="margin-top: 3rem;"></div>', unsafe_allow_html=True)

    # CTA Button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Build Your Portfolio", use_container_width=True, type="primary", key="start_btn"):
            st.switch_page("pages/1_Stock_Selection.py")

    # Footer
    st.markdown("""
    <div class="landing-footer">
        <p style="font-size: 0.9rem;">Quantitative Asset and Risk Management II | University of Lausanne</p>
        <p style="font-size: 0.8rem; margin-top: 0.3rem;">Salihu Viktor | Misini Erion | Cavard Augustin | Aslan Iltan</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
