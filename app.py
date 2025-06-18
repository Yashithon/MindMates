import streamlit as st
import joblib
import random
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Mental Wellness Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load model and vectorizer
@st.cache_resource
def load_models():
    try:
        model = joblib.load("mental_health_model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'mental_health_model.pkl' and 'vectorizer.pkl' are in the correct directory.")
        return None, None

model, vectorizer = load_models()

# Motivational quotes for different mental health categories
QUOTES = {
    "anxiety": [
        "Anxiety is not your enemy. It's your body trying to protect you. Be gentle with yourself.",
        "You are stronger than your anxiety. One breath at a time, one step at a time.",
        "Your anxiety does not define you. You are capable of amazing things.",
        "It's okay to feel anxious. What matters is how you respond to it with kindness."
    ],
    "depression": [
        "Even the darkest night will end and the sun will rise. You are not alone.",
        "Your mental health is just as important as your physical health. Take care of both.",
        "Every small step forward is progress. Be proud of how far you've come.",
        "You matter. Your story matters. Your life has meaning and purpose."
    ],
    "stress": [
        "Stress is temporary, but your strength is permanent. You've got this.",
        "Take time to rest. It's not a luxury, it's a necessity.",
        "You don't have to carry the weight of the world. Set it down and breathe.",
        "Progress over perfection. You're doing better than you think."
    ],
    "bipolar": [
        "Your ups and downs don't define your worth. You are valued in every season.",
        "Stability comes with time, patience, and self-compassion. Keep going.",
        "You are not your diagnosis. You are a whole person with many beautiful qualities.",
        "Every day you choose to keep going is an act of courage."
    ],
    "ptsd": [
        "Healing isn't linear, and that's okay. Be patient with your journey.",
        "You survived something difficult. That makes you incredibly strong.",
        "Your past does not determine your future. You have the power to heal.",
        "Seeking help is not weakness; it's wisdom and courage."
    ],
    "eating_disorder": [
        "Your worth is not measured by numbers. You are valuable exactly as you are.",
        "Recovery is possible. Take it one meal, one day at a time.",
        "Your body deserves nourishment and kindness, not punishment.",
        "You are more than your relationship with food. You are worthy of love."
    ],
    "addiction": [
        "Recovery is a journey, not a destination. Every step counts.",
        "You have the strength to overcome this. Believe in your power to heal.",
        "One day at a time. One choice at a time. You can do this.",
        "Your past mistakes don't define your future possibilities."
    ],
    "default": [
        "Mental health is not a destination, but a process. Be kind to yourself along the way.",
        "It's okay not to be okay. What matters is that you're here and you're trying.",
        "You are braver than you believe, stronger than you seem, and more loved than you know.",
        "Taking care of your mental health is an act of self-love."
    ]
}

# Enhanced CSS with mint green color scheme
st.markdown("""
<style>
    /* Import calming fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    
    /* CSS Variables for mint green color scheme */
    :root {
        --mint-light: #EBFFF5;
        --mint-medium: #CFECE0;
        --mint-dark: #7CAE9E;
        --cream: #FFFEF7;
        --cream-light: #FEFCF3;
        --text-primary: #2D5A4A;
        --text-secondary: #5A8471;
        --shadow-light: rgba(124, 174, 158, 0.15);
        --shadow-medium: rgba(124, 174, 158, 0.25);
    }
    
    /* Global styling with mint color scheme */
    .stApp {
        background: linear-gradient(135deg, var(--cream) 0%, var(--mint-light) 50%, var(--cream-light) 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main > div {
        padding-top: 1rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }
    
    /* Header section with mint gradient */
    .title-container {
        text-align: center;
        padding: 4rem 2rem;
        background: linear-gradient(135deg, var(--mint-light) 0%, var(--cream) 100%);
        border-radius: 35px;
        margin: 2rem 0 3rem 0;
        box-shadow: 0 10px 40px var(--shadow-light);
        border: 1px solid var(--mint-medium);
    }
    
    .title-container h1 {
        font-size: 3rem;
        margin-bottom: 1rem;
        font-weight: 600;
        letter-spacing: -1px;
        color: var(--text-primary);
        line-height: 1.2;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: var(--text-secondary);
        margin: 0;
        font-weight: 300;
        line-height: 1.5;
        max-width: 600px;
        margin: 0 auto;
    }
    
    /* Input container with cream background */
    .input-container {
        background: var(--cream);
        padding: 3rem 2.5rem;
        border-radius: 25px;
        box-shadow: 0 8px 30px var(--shadow-light);
        margin-bottom: 2rem;
        border: 1px solid var(--mint-medium);
    }
    
    .input-container h3 {
        color: var(--text-primary) !important;
        font-size: 1.5rem !important;
        margin-bottom: 1rem !important;
        font-weight: 500 !important;
    }
    
    .input-description {
        color: var(--text-secondary);
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    /* Results containers with mint theme */
    .result-container {
        background: linear-gradient(135deg, var(--mint-light) 0%, var(--cream) 100%);
        padding: 3rem 2.5rem;
        border-radius: 25px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 30px var(--shadow-light);
        border: 2px solid var(--mint-medium);
        animation: slideInUp 0.6s ease-out;
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .quote-container {
        background: linear-gradient(135deg, var(--cream-light) 0%, var(--cream) 100%);
        padding: 3rem 2.5rem;
        border-radius: 25px;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 30px var(--shadow-light);
        border: 1px solid var(--mint-light);
        animation: slideInUp 0.8s ease-out;
    }
    
    .prediction-tag {
        font-size: 2rem;
        font-weight: 600;
        color: var(--mint-dark);
        margin-bottom: 1rem;
        text-transform: capitalize;
        letter-spacing: 0.3px;
    }
    
    .result-subtitle {
        color: var(--text-secondary);
        font-size: 1rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    .quote-text {
        font-size: 1.3rem;
        font-style: italic;
        color: var(--text-primary);
        line-height: 1.7;
        margin-bottom: 1rem;
        font-weight: 300;
    }
    
    .quote-subtitle {
        color: var(--text-secondary);
        font-size: 1rem;
        font-weight: 400;
    }
    
    /* Encouragement box with mint theme */
    .encouragement {
        background: linear-gradient(135deg, var(--mint-light) 0%, var(--cream-light) 100%);
        padding: 3rem 2rem;
        border-radius: 25px;
        text-align: center;
        margin: 3rem 0;
        color: var(--text-primary);
        font-weight: 300;
        line-height: 1.7;
        font-size: 1.1rem;
        box-shadow: 0 8px 30px var(--shadow-light);
        border: 1px solid var(--mint-medium);
    }
    
    .encouragement h3 {
        color: var(--text-primary);
        font-size: 1.5rem;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    
    /* Stats section */
    .stats-section {
        margin: 3rem 0;
        padding: 0 1rem;
    }
    
    .stats-title {
        text-align: center;
        color: var(--text-primary);
        font-size: 1.5rem;
        font-weight: 500;
        margin-bottom: 2rem;
        letter-spacing: 0.3px;
    }
    
    .stat-box {
        background: var(--cream);
        padding: 2rem 1.5rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 5px 20px var(--shadow-light);
        border: 1px solid var(--mint-light);
        transition: all 0.3s ease;
    }
    
    .stat-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px var(--shadow-medium);
    }
    
    .stat-number {
        font-size: 2rem;
        font-weight: 600;
        color: var(--mint-dark);
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        color: var(--text-secondary);
        font-size: 0.9rem;
        font-weight: 400;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Button styling with mint gradient */
    .stButton > button {
        background: linear-gradient(135deg, var(--mint-dark) 0%, var(--mint-medium) 100%);
        color: white;
        border: none;
        padding: 1.2rem 2.5rem;
        border-radius: 50px;
        font-size: 1.1rem;
        font-weight: 500;
        box-shadow: 0 6px 20px var(--shadow-medium);
        transition: all 0.3s ease;
        width: 100%;
        letter-spacing: 0.5px;
        font-family: 'Inter', sans-serif;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px var(--shadow-medium);
        background: linear-gradient(135deg, var(--mint-dark) 0%, #6BA490 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Textarea styling */
    .stTextArea > div > div > textarea {
        border-radius: 18px;
        border: 2px solid var(--mint-medium);
        padding: 1.5rem;
        font-size: 1rem;
        min-height: 150px;
        background: linear-gradient(135deg, var(--cream-light) 0%, var(--cream) 100%);
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
        line-height: 1.6;
        resize: vertical;
        transition: all 0.3s ease;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: var(--mint-dark);
        box-shadow: 0 0 0 4px var(--shadow-light);
        background: var(--cream);
        outline: none;
    }
    
    .stTextArea > div > div > textarea::placeholder {
        color: var(--text-secondary);
        font-weight: 300;
    }
    
    /* Information cards */
    .info-section {
        margin: 3rem 0;
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 2rem;
    }
    
    .info-card {
        background: var(--cream-light);
        padding: 2rem;
        border-radius: 20px;
        border: 1px solid var(--mint-light);
        box-shadow: 0 5px 20px var(--shadow-light);
        transition: all 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px var(--shadow-medium);
    }
    
    .info-card h4 {
        color: var(--text-primary);
        font-size: 1.2rem;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    
    .info-card p, .info-card li {
        color: var(--text-secondary);
        font-size: 0.9rem;
        line-height: 1.6;
        font-weight: 300;
    }
    
    .info-card ul {
        list-style: none;
        padding: 0;
    }
    
    .info-card li {
        margin-bottom: 0.5rem;
        padding-left: 1.5rem;
        position: relative;
    }
    
    .info-card li::before {
        content: "•";
        color: var(--mint-dark);
        position: absolute;
        left: 0;
        font-weight: bold;
    }
    
    /* Expandable sections */
    .stExpander {
        background: var(--cream-light);
        border-radius: 15px;
        border: 1px solid var(--mint-light);
        overflow: hidden;
        box-shadow: 0 4px 15px var(--shadow-light);
    }
    
    .stExpander > div > div {
        background: var(--cream-light);
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 3rem 2rem;
        color: white;
        margin-top: 4rem;
        background: linear-gradient(135deg, var(--mint-medium) 0%, var(--mint-dark) 100%);
        border-radius: 25px;
        line-height: 1.6;
        box-shadow: 0 8px 30px var(--shadow-light);
    }
    
    .footer-main {
        font-size: 1rem;
        margin-bottom: 1rem;
        font-weight: 300;
    }
    
    .footer-disclaimer {
        font-size: 0.9rem;
        opacity: 0.8;
        font-weight: 300;
    }
    
    /* Header overrides */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-top-color: var(--mint-dark) !important;
    }
    
    /* Success/Info messages */
    .stSuccess {
        background: linear-gradient(135deg, var(--mint-light) 0%, var(--cream) 100%);
        border-radius: 15px;
        border: 1px solid var(--mint-medium);
    }
    
    .stInfo {
        background: linear-gradient(135deg, var(--cream-light) 0%, var(--cream) 100%);
        border-radius: 15px;
        border: 1px solid var(--mint-light);
    }
    
    /* Divider styling */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, var(--mint-medium), transparent);
        margin: 3rem 0;
    }
    
    /* Results anchor for smooth scrolling */
    .results-anchor {
        scroll-margin-top: 100px;
    }
    
    /* Column spacing */
    .element-container {
        margin-bottom: 1rem;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .title-container h1 {
            font-size: 2rem;
        }
        
        .subtitle {
            font-size: 1rem;
        }
        
        .input-container {
            padding: 2rem 1.5rem;
        }
        
        .result-container, .quote-container {
            padding: 2rem 1.5rem;
        }
        
        .prediction-tag {
            font-size: 1.5rem;
        }
        
        .quote-text {
            font-size: 1.1rem;
        }
        
        .info-section {
            grid-template-columns: 1fr;
        }
    }
</style>

<script>
// Smooth scroll to results when button is clicked
function scrollToResults() {
    // Wait a bit for Streamlit to render the results
    setTimeout(function() {
        const resultsSection = document.querySelector('.results-anchor');
        if (resultsSection) {
            resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    }, 100);
}

// Add event listener to the analyze button
document.addEventListener('DOMContentLoaded', function() {
    // Monitor for button clicks
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.type === 'childList') {
                const buttons = document.querySelectorAll('.stButton button');
                buttons.forEach(function(button) {
                    if (button.textContent.includes('Understand My Thoughts') && !button.hasAttribute('data-listener')) {
                        button.setAttribute('data-listener', 'true');
                        button.addEventListener('click', function() {
                            setTimeout(scrollToResults, 500);
                        });
                    }
                });
            }
        });
    });
    
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
});
</script>
""", unsafe_allow_html=True)

# Prediction function
def predict_tag(text):
    if model is None or vectorizer is None:
        return "error"
    
    text = text.lower()
    text_vec = vectorizer.transform([text])
    return model.predict(text_vec)[0]

def get_random_quote(tag):
    tag_lower = tag.lower()
    if tag_lower in QUOTES:
        return random.choice(QUOTES[tag_lower])
    return random.choice(QUOTES["default"])

# Main UI
st.markdown("""
<style>
.subtitle {
    text-align: center;
    margin-top: 0.5rem;
    font-size: 1.1rem;
    color: #555;
}
</style>

<div class="title-container">
    <h1>🌿 Welcome to MindMates</h1>

</div>
""", unsafe_allow_html=True)


# Initialize session state for stats
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = 0
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

# Main content layout
col1, col2 = st.columns([2, 1])

with col1:
    # st.markdown('<div class="input-container">', unsafe_allow_html=True)
    st.markdown("### Share What's on Your Mind")
    st.markdown('<p class="input-description">This is a safe space to express your thoughts and feelings. MindMates will help provide context about what you\'re experiencing. Remember, this is for awareness and support—professional guidance is always recommended for your wellbeing.</p>', unsafe_allow_html=True)
    
    user_input = st.text_area(
        "",
        placeholder="Take your time... you might write something like: 'I've been feeling overwhelmed lately and find it hard to concentrate on daily tasks'",
        height=150,
        help="Share as much or as little as feels comfortable. There's no pressure—just gentle understanding.",
        key="user_input_text"
    )
    
    predict_button = st.button("🌸 Understand My Thoughts", type="primary", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    # Info cards in sidebar
    st.markdown("""
    <div class="info-card">
        <h4>🌱 How This Helps</h4>
        <ul>
            <li>Provides gentle insights into mental health topics</li>
            <li>This response is generated by a machine learning model and is not a medical diagnosis.</li>
            <li>Supports self-reflection and awareness</li>
            <li>Creates a safe space for expression</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h4>🛡️ Important Reminders</h4>
        <ul>
            <li>Always seek professional help for proper care</li>
            <li>Contact emergency services if you're in crisis</li>
            <li>Your wellbeing matters and help is available</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Results section with anchor for smooth scrolling
if predict_button and user_input.strip():
    st.markdown('<div class="results-anchor"></div>', unsafe_allow_html=True)
    
    if model is not None and vectorizer is not None:
        with st.spinner("🌿 Gently analyzing your words..."):
            result = predict_tag(user_input)
            quote = get_random_quote(result)
            
            # Update session state
            st.session_state.predictions_made += 1
            st.session_state.last_prediction = result
            
            # Display results in two columns
            result_col1, result_col2 = st.columns([1, 1])
            
            with result_col1:
                st.markdown(f"""
                <div class="result-container">
                    <h2>Understanding Your Experience</h2>
                    <div class="prediction-tag">{result.replace('_', ' ').title()}</div>
                    <p class="result-subtitle">This reflection may help you better understand what you're going through</p>
                </div>
                """, unsafe_allow_html=True)
            
            with result_col2:
                st.markdown(f"""
                <div class="quote-container">
                    <h3>A Gentle Reminder</h3>
                    <p class="quote-text">"{quote}"</p>
                    <p class="quote-subtitle">You are taking positive steps by seeking understanding</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Encouragement message
            st.markdown("""
            <div class="encouragement">
                <h3>🌸 You showed courage by expressing your thoughts today</h3>
                <p>Self-awareness and reflection are meaningful steps toward wellness and personal growth. Remember, seeking understanding is a sign of strength, not weakness.</p>
            </div>
            """, unsafe_allow_html=True)
            
    else:
        st.error("Unable to load the classification model. Please check that the model files are available.")

elif predict_button and not user_input.strip():
    st.info("Please share some thoughts when you're ready to explore them.")

# Stats section
if st.session_state.predictions_made > 0:
    st.markdown("---")
    st.markdown('<div class="stats-section">', unsafe_allow_html=True)
    st.markdown('<h3 class="stats-title">Your Reflection Journey</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{st.session_state.predictions_made}</div>
            <div class="stat-label">Reflections Shared</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        latest_display = st.session_state.last_prediction.replace('_', ' ').title() if st.session_state.last_prediction else "None yet"
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number" style="font-size: 1.2rem;">{latest_display}</div>
            <div class="stat-label">Recent Understanding</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        current_time = datetime.now().strftime("%I:%M %p")
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-number">{current_time}</div>
            <div class="stat-label">Current Time</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Information section
st.markdown("---")

# Resources section
with st.expander("🌿 Helpful Mental Health Resources", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Crisis Support (India):**
        - Vandrevala Foundation: 9999 666 555 (Available 24/7)
        - AASRA: 91-22-27546669 (Available 24/7)
        - Sneha Foundation: 044-24640050 (Available 24/7)
        - iCall: 9152987821 (Monday-Saturday, 8am-10pm)
        - Samaritans Mumbai: +91 84229 84528 (Available 24/7)

        **Professional Support:**
        - Practo: Find qualified therapists and psychiatrists nearby
        - BetterLYF: Professional online counseling services
        - YourDOST: Comprehensive mental health support platform
        - Fortis Healthcare Mental Health Services
        """)
    
    with col2:
        st.markdown("""
        **Supportive Apps & Tools:**
        - Wysa: AI companion for mental health support
        - InnerHour: Evidence-based mental wellness approach
        - MindShift: Tools for managing anxiety
        - Headspace: Guided meditation and mindfulness
        
        **Self-Care Resources:**
        - Mindfulness meditation techniques
        - Breathing exercises and relaxation methods
        - Journaling for mental clarity
        - Physical exercise and outdoor activities
        - Creative expression and art therapy
        """)

# Footer
st.markdown("""
<div class="footer">
    <p class="footer-main">Created with care and compassion for mental health awareness and support </p>
    <p class="footer-disclaimer">
       This tool is a basic educational resource created as a personal project. It is not intended for medical diagnosis or treatment. Please consult a qualified healthcare professional for medical advice.
    </p>
</div>
""", unsafe_allow_html=True)

# Add breathing space
st.markdown("<br>", unsafe_allow_html=True)
