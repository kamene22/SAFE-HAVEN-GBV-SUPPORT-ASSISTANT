import streamlit as st
import pandas as pd
import torch
import requests
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, AutoModelForCausalLM
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="GBV Support Assistant",
    page_icon="💜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background-color: #6a0dad;
        color: white;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .ribbon-container {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .gbv-ribbon {
        width: 60px;
        height: 60px;
        background-color: #ff69b4;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        color: white;
        font-size: 24px;
        font-weight: bold;
    }
    
    .user-message {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 1rem 1rem 0 1rem;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
    }
    
    .assistant-message {
        background-color: #6a0dad;
        color: white;
        padding: 1rem;
        border-radius: 1rem 1rem 1rem 0;
        margin: 0.5rem 0;
        max-width: 80%;
    }
    
    .stButton button {
        background-color: #6a0dad;
        color: white;
    }
    
    .quick-exit {
        background-color: #ff4d4d !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
        border-radius: 0.5rem 0.5rem 0 0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #6a0dad;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Translations dictionary
TRANSLATIONS = {
    'en': {
        'title': "GBV Support Assistant",
        'subtitle': "A safe space for gender-based violence survivors",
        'language': "Switch to Swahili",
        'quick_exit': "Quick Exit",
        'loading': "Loading support resources...",
        'chat_placeholder': "Type your message here...",
        'send': "Send",
        'clear': "Clear Conversation",
        'report_incident': "Report Incident",
        'emergency_contacts': "Emergency Contacts",
        'safety_planning': "Safety Planning",
        'resources': "Resources",
        'privacy_notice': "Privacy Notice",
        'incident_type': "Incident Type",
        'incident_date': "Incident Date",
        'incident_location': "Incident Location",
        'incident_description': "Description (optional)",
        'submit_report': "Submit Report",
        'report_confirmation': "Thank you for your report. It has been received confidentially.",
        'safety_tips_title': "Safety Planning Tips",
        'safety_tips': [
            "Identify safe places to go in an emergency",
            "Keep important documents and emergency money in a safe place",
            "Memorize important phone numbers",
            "Establish a code word with trusted friends/family to signal danger",
            "Plan escape routes from your home"
        ],
        'privacy_content': "All conversations are confidential. Your safety is our priority. We don't store personal identifiers.",
        'welcome_message': "Hello, I'm here to help. How can I support you today?",
        'search_resources': "Search Resources",
        'select_county': "Select County",
        'search': "Search",
        'shelters_found': "Shelters found in {}",
        'hospitals_found': "Hospitals found in {}",
        'name': "Name",
        'contact': "Contact",
        'services': "Services",
        'type': "Type",
        'no_results': "No results found for {}",
        'filter_services': "Filter by Services",
        'all_services': "All Services",
        'medical': "Medical",
        'counseling': "Counseling",
        'legal': "Legal",
        'shelter': "Shelter",
        'emergency': "Emergency",
        'other': "Other",
        'ask_ai': "Ask AI Assistant",
        'ai_placeholder': "Ask the AI anything about GBV support...",
        'ai_thinking': "AI is thinking...",
        'api_error': "Error connecting to AI service"
    },
    'sw': {
        'title': "Msaidizi wa Utekaji Nyara Kijinsia",
        'subtitle': "Eneo salama kwa wahasiriwa wa ukatili wa kijinsia",
        'language': "Badilisha lugha (English)",
        'quick_exit': "Ondoka Haraka",
        'loading': "Inapakia rasilimali za usaidizi...",
        'chat_placeholder': "Andika ujumbe wako hapa...",
        'send': "Tuma",
        'clear': "Futa Mazungumzo",
        'report_incident': "Ripoti Tukio",
        'emergency_contacts': "Mawasiliano ya Dharura",
        'safety_planning': "Mipango ya Usalama",
        'resources': "Rasilimali",
        'privacy_notice': "Arifa ya Faragha",
        'incident_type': "Aina ya Tukio",
        'incident_date': "Tarehe ya Tukio",
        'incident_location': "Mahali pa Tukio",
        'incident_description': "Maelezo (ya hiari)",
        'submit_report': "Wasilisha Ripoti",
        'report_confirmation': "Asante kwa ripoti yako. Imepokelewa kwa siri.",
        'safety_tips_title': "Vidokezo vya Usalama",
        'safety_tips': [
            "Tambua sehemu salama za kwenda wakati wa dharura",
            "Hifadhi hati muhimu na pesa za dharura mahali salama",
            "Kumbuka nambari muhimu za simu",
            "Weka neno la siri na rafiki/jamaa unaowaamini kuashiria hatari",
            "Panga njia za kutoroka kutoka nyumbani kwako"
        ],
        'privacy_content': "Mazungumzo yote ni ya siri. Usalama wako ni kipaumbele chetu. Hatuwezi kuhifadhi taarifa za kutambulisha.",
        'welcome_message': "Hujambo, niko hapa kukusaidia. Naweza kukusaidia vipi leo?",
        'search_resources': "Tafuta Rasilimali",
        'select_county': "Chagua Kaunti",
        'search': "Tafuta",
        'shelters_found': "Makazi ya wakati mfupi yaliyopatikana {}",
        'hospitals_found': "Hospitali zilizopatikana {}",
        'name': "Jina",
        'contact': "Mawasiliano",
        'services': "Huduma",
        'type': "Aina",
        'no_results': "Hakuna matokeo ya {}",
        'filter_services': "Chuja kwa Huduma",
        'all_services': "Huduma Zote",
        'medical': "Matibabu",
        'counseling': "Ushauri",
        'legal': "Kisheria",
        'shelter': "Makazi",
        'emergency': "Dharura",
        'other': "Nyingine",
        'ask_ai': "Uliza Msaidizi wa AI",
        'ai_placeholder': "Uliza AI chochote kuhusu usaidizi wa GBV...",
        'ai_thinking': "AI inafikiria...",
        'api_error': "Hitilafu ya kuungana na huduma ya AI"
    }
}

def create_gbv_ribbon():
    """Create GBV awareness ribbon icon"""
    return """
    <div class="gbv-ribbon">
        <span>GBV</span>
    </div>
    """

def load_help_center_data():
    """Load help center data including hospitals and shelters"""
    try:
        with open('gbv_resources.json', 'r') as f:
            data = json.load(f)
            
        if 'service_types' not in data:
            data['service_types'] = {
                'en': ["Medical", "Counseling", "Legal", "Shelter", "Emergency", "Other"],
                'sw': ["Matibabu", "Ushauri", "Kisheria", "Makazi", "Dharura", "Nyingine"]
            }
            
    except:
        data = {
            'counties': ['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Uasin Gishu'],
            'service_types': {
                'en': ["Medical", "Counseling", "Legal", "Shelter", "Emergency", "Other"],
                'sw': ["Matibabu", "Ushauri", "Kisheria", "Makazi", "Dharura", "Nyingine"]
            },
            'resources': [
                {
                    "name": "Nairobi Women's Hospital",
                    "county": "Nairobi",
                    "type": "hospital",
                    "contact": "020-1234567",
                    "services": "GBV care, counseling",
                    "service_types": ["Medical", "Counseling"]
                },
                {
                    "name": "Coast General Hospital",
                    "county": "Mombasa",
                    "type": "hospital",
                    "contact": "041-7654321",
                    "services": "Emergency GBV services",
                    "service_types": ["Medical", "Emergency"]
                },
                {
                    "name": "Nairobi Women's Shelter",
                    "county": "Nairobi",
                    "type": "shelter",
                    "contact": "0722123456",
                    "services": "Temporary housing, counseling",
                    "service_types": ["Shelter", "Counseling"]
                }
            ]
        }
    return data

# Initialize session state
if 'language' not in st.session_state:
    st.session_state.language = 'en'
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'report_submitted' not in st.session_state:
    st.session_state.report_submitted = False

def get_text(key):
    """Get translated text"""
    return TRANSLATIONS[st.session_state.language].get(key, key)

def load_models():
    """Load AI models with progress tracking"""
    if st.session_state.model_loaded:
        return st.session_state.models
    
    with st.spinner(get_text('loading')):
        progress_bar = st.progress(0)
        
        try:
            # Load XLMRoberta model (intent classification)
            progress_bar.progress(25)
            xlm_model_path = "xlm-roberta-base"
            xlm_tokenizer = AutoTokenizer.from_pretrained(xlm_model_path)
            xlm_model = AutoModelForSequenceClassification.from_pretrained(xlm_model_path, num_labels=7)
            xlm_model.eval()
            
            # Load TinyLlama model (chatbot)
            progress_bar.progress(50)
            llama_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_id)
            llama_model = AutoModelForCausalLM.from_pretrained(llama_model_id, device_map="auto")
            chatbot = pipeline("text-generation", model=llama_model, tokenizer=llama_tokenizer)
            
            # Load help center data
            progress_bar.progress(75)
            help_center_data = load_help_center_data()
            
            progress_bar.progress(100)
            
            models = {
                'xlm_tokenizer': xlm_tokenizer,
                'xlm_model': xlm_model,
                'llama_tokenizer': llama_tokenizer,
                'llama_model': llama_model,
                'chatbot': chatbot,
                'help_center_data': help_center_data
            }
            
            st.session_state.models = models
            st.session_state.model_loaded = True
            progress_bar.empty()
            
            return models
            
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return None

def classify_intent(text):
    """Classify user intent using XLMRoberta model"""
    if not st.session_state.model_loaded:
        return "unknown"
    
    try:
        models = st.session_state.models
        tokenizer = models['xlm_tokenizer']
        model = models['xlm_model']
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        intent_id = torch.argmax(predictions, dim=-1).item()
        
        intent_labels = [
            "find_shelter", "get_hotline", "legal_help", 
            "report_gbv", "ask_definition", "exit", "unknown"
        ]
        
        return intent_labels[intent_id]
    except Exception as e:
        return "unknown"

def generate_chat_response(prompt, context=None):
    """Generate response using TinyLlama model"""
    if not st.session_state.model_loaded:
        return "I'm having trouble generating a response. Please try again."
    
    try:
        chatbot = st.session_state.models['chatbot']
        
        if context:
            full_prompt = f"Context: {context}\n\nUser: {prompt}\nAssistant:"
        else:
            full_prompt = f"User: {prompt}\nAssistant:"
        
        response = chatbot(
            full_prompt,
            max_length=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1
        )
        
        generated_text = response[0]['generated_text']
        assistant_response = generated_text.replace(full_prompt, "").strip()
        
        return assistant_response
    except Exception as e:
        return f"I encountered an error: {str(e)}"

# API Configuration
API_CONFIG = {
    'url': "https://api.deepseek.com/v1/chat/completions",
    'headers': {
        'Authorization': f"Bearer {st.secrets['sk-7ef25e599659474e8c7a2fe863cf7226']}",
        'Content-Type': "application/json"
    }
}


def query_ai_api(prompt, context=None):
    """Send query to AI API and return response"""
    try:
        payload = {
            "prompt": prompt,
            "context": context or "",
            "language": st.session_state.language,
            "max_tokens": 200,
            "temperature": 0.7
        }
        
        with st.spinner(get_text('ai_thinking')):
            response = requests.post(
                API_CONFIG['url'],
                headers=API_CONFIG['headers'],
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json().get('response', get_text('api_error'))
            else:
                return get_text('api_error')
                
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return get_text('api_error')

def generate_shelter_response(help_data):
    """Generate response for shelter request"""
    shelters = help_data['shelters'].get(st.session_state.language, [])
    
    if not shelters:
        return get_text('no_shelters')
    
    response = f"Here are some shelters that might help:\n\n"
    for shelter in shelters:
        response += f"• {shelter['name']} ({shelter['location']})\n"
        response += f"  Contact: {shelter['contact']}\n"
        response += f"  Services: {shelter['services']}\n\n"
    
    return response

def generate_contact_response(help_data):
    """Generate response for emergency contacts request"""
    hotlines = help_data['hotlines'].get(st.session_state.language, [])
    
    if not hotlines:
        return get_text('no_hotlines')
    
    response = f"Here are emergency contacts:\n\n"
    for contact in hotlines:
        response += f"• {contact['name']}: {contact['number']}\n"
    
    response += "\nYou can call these numbers anytime for help."
    return response

def generate_legal_response(help_data):
    """Generate response for legal help request"""
    legal_aid = help_data['legal_aid'].get(st.session_state.language, [])
    
    if not legal_aid:
        return get_text('no_legal')
    
    response = f"Here are legal aid organizations:\n\n"
    for org in legal_aid:
        response += f"• {org['name']}\n"
        response += f"  Contact: {org['contact']}\n"
        response += f"  Services: {org['services']}\n\n"
    
    return response

def generate_response(user_message):
    """Generate contextual response based on intent and content"""
    intent = classify_intent(user_message)
    help_data = st.session_state.models.get('help_center_data', {})
    
    if intent == "find_shelter":
        return generate_shelter_response(help_data)
    elif intent == "get_hotline":
        return generate_contact_response(help_data)
    elif intent == "legal_help":
        return generate_legal_response(help_data)
    elif intent == "report_gbv":
        return """If you'd like to report a GBV incident, please use the "Report Incident" 
        section in the sidebar. Your report will be handled confidentially."""
    elif intent == "ask_definition":
        return """Gender-Based Violence (GBV) refers to harmful acts directed at an individual 
        based on their gender. It includes physical, sexual, emotional, psychological, and 
        economic abuse."""
    
    context = """You are a compassionate support assistant for gender-based violence survivors. 
    Provide empathetic, non-judgmental responses. Offer practical help when appropriate. 
    Always prioritize safety and confidentiality."""
    
    return generate_chat_response(user_message, context)

def display_chat():
    """Display chat messages"""
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"""
                <div class="user-message">
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="assistant-message">
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)

def report_incident_form():
    """Display incident reporting form in sidebar"""
    with st.sidebar:
        st.subheader(get_text('report_incident'))
        
        with st.form("incident_report"):
            incident_type = st.selectbox(
                get_text('incident_type'),
                ["Physical violence", "Sexual violence", "Emotional abuse", "Economic abuse", "Other"]
            )
            
            incident_date = st.date_input(get_text('incident_date'))
            incident_location = st.text_input(get_text('incident_location'))
            description = st.text_area(get_text('incident_description'))
            
            submitted = st.form_submit_button(get_text('submit_report'))
            
            if submitted:
                st.success(get_text('report_confirmation'))
                st.session_state.report_submitted = True

def display_resource_results(resources, county):
    """Display filtered resources in a tabbed view"""
    tab1, tab2 = st.tabs([
        f"🏥 {get_text('hospitals_found').format(county)}",
        f"🏠 {get_text('shelters_found').format(county)}"
    ])
    
    hospitals = [r for r in resources if r['type'] == 'hospital']
    shelters = [r for r in resources if r['type'] == 'shelter']
    
    with tab1:
        if hospitals:
            hospital_df = pd.DataFrame(hospitals)[['name', 'contact', 'services', 'service_types']]
            st.dataframe(
                hospital_df,
                column_config={
                    'name': get_text('name'),
                    'contact': get_text('contact'),
                    'services': get_text('services'),
                    'service_types': get_text('type')
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info(f"No hospitals found in {county} with selected filters")
    
    with tab2:
        if shelters:
            shelter_df = pd.DataFrame(shelters)[['name', 'contact', 'services', 'service_types']]
            st.dataframe(
                shelter_df,
                column_config={
                    'name': get_text('name'),
                    'contact': get_text('contact'),
                    'services': get_text('services'),
                    'service_types': get_text('type')
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info(f"No shelters found in {county} with selected filters")

def display_resource_search():
    """Display resource search by county with service filtering"""
    with st.sidebar:
        st.subheader(get_text('search_resources'))
        
        help_data = st.session_state.models.get('help_center_data', {})
        counties = help_data.get('counties', [])
        service_types = help_data.get('service_types', {}).get(st.session_state.language, [])
        
        selected_county = st.selectbox(
            get_text('select_county'),
            counties,
            key='county_select'
        )
        
        selected_services = st.multiselect(
            get_text('filter_services'),
            options=[get_text('all_services')] + service_types,
            default=[get_text('all_services')],
            key='service_filter'
        )
        
        if st.button(get_text('search')):
            resources = help_data.get('resources', [])
            county_resources = [r for r in resources if r['county'].lower() == selected_county.lower()]
            
            if not county_resources:
                st.error(get_text('no_results').format(selected_county))
                return
            
            if get_text('all_services') not in selected_services:
                reverse_trans = {v: k for k, v in TRANSLATIONS[st.session_state.language].items()}
                english_services = [reverse_trans.get(s, s) for s in selected_services]
                
                filtered_resources = []
                for resource in county_resources:
                    resource_services = resource.get('service_types', [])
                    if any(service in resource_services for service in english_services):
                        filtered_resources.append(resource)
                county_resources = filtered_resources
            
            if not county_resources:
                st.error(get_text('no_results').format(selected_county))
                return
            
            display_resource_results(county_resources, selected_county)

def display_ai_button():
    """Display the Ask AI button in sidebar"""
    with st.sidebar:
        st.subheader(get_text('ask_ai'))
        ai_prompt = st.text_area(
            get_text('ai_placeholder'),
            key='ai_prompt',
            height=100
        )
        
        if st.button("✨ " + get_text('ask_ai')):
            if ai_prompt.strip():
                context = "\n".join(
                    [msg['content'] for msg in st.session_state.chat_history[-3:]]
                )
                
                ai_response = query_ai_api(ai_prompt, context)
                
                st.session_state.chat_history.extend([
                    {'role': 'user', 'content': ai_prompt},
                    {'role': 'assistant', 'content': ai_response}
                ])
                st.rerun()
            else:
                st.warning("Please enter a question for the AI")

def display_resources():
    """Display resources in sidebar"""
    with st.sidebar:
        st.subheader(get_text('resources'))
        
        if st.button(get_text('emergency_contacts')):
            response = generate_contact_response(st.session_state.models['help_center_data'])
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response
            })
            st.rerun()
            
        if st.button(get_text('safety_planning')):
            tips = TRANSLATIONS[st.session_state.language]['safety_tips']
            response = f"{get_text('safety_tips_title')}:\n\n" + "\n\n• ".join(tips)
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response
            })
            st.rerun()
            
        st.subheader(get_text('privacy_notice'))
        st.info(get_text('privacy_content'))

def main():
    """Main application function"""
    # Language toggle and quick exit buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    with col3:
        if st.button("🌐 " + get_text('language')):
            st.session_state.language = 'sw' if st.session_state.language == 'en' else 'en'
            st.rerun()
        
        if st.button("🚪 " + get_text('quick_exit'), type="primary"):
            st.markdown("""
            <script>
                window.open('https://www.google.com', '_blank');
            </script>
            """, unsafe_allow_html=True)
    
    # Header with ribbon
    st.markdown(f"""
    <div class="main-header">
        <div class="ribbon-container">
            {create_gbv_ribbon()}
            <div>
                <h1>{get_text('title')}</h1>
                <p>{get_text('subtitle')}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    models = load_models()
    if not models:
        st.error("Failed to load models. Please refresh the page.")
        return
    
    # Sidebar content
    report_incident_form()
    display_resource_search()
    display_ai_button()
    display_resources()
    
    # Chat interface
    display_chat()
    
    # Add welcome message if chat is empty
    if not st.session_state.chat_history:
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': get_text('welcome_message')
        })
        st.rerun()
    
    # Chat input
    chat_input = st.chat_input(get_text('chat_placeholder'))
    
    if chat_input:
        st.session_state.chat_history.append({
            'role': 'user',
            'content': chat_input
        })
        
        response = generate_response(chat_input)
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': response
        })
        
        st.rerun()
    
    # Clear conversation button
    if st.button(get_text('clear')):
        st.session_state.chat_history = [{
            'role': 'assistant',
            'content': get_text('welcome_message')
        }]
        st.rerun()

if __name__ == "__main__":
    main()