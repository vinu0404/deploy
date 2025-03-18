import streamlit as st
import boto3
import os
import json
from botocore.exceptions import BotoCoreError, NoCredentialsError
from langchain_aws import BedrockLLM
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import BedrockEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Fetch AWS credentials securely from Render environment variables
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_SECRET_NAME = os.getenv("AWS_SECRET_NAME")

# Function to fetch secrets from AWS Secrets Manager
def get_aws_secrets(secret_name, region_name="us-east-1"):
    try:
        session = boto3.session.Session()
        client = session.client(service_name="secretsmanager", region_name=region_name)
        secret_response = client.get_secret_value(SecretId=secret_name)
        secret = json.loads(secret_response["SecretString"])
        return secret
    except (BotoCoreError, NoCredentialsError) as e:
        st.error(f"Error fetching AWS Secrets: {e}")
        return None

# Retrieve secrets
aws_secrets = get_aws_secrets(secret_name=AWS_SECRET_NAME, region_name=AWS_REGION)

if aws_secrets:
    bedrock_client = boto3.client(
        service_name="bedrock-runtime",
        region_name=AWS_REGION,
        aws_access_key_id=aws_secrets["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=aws_secrets["AWS_SECRET_ACCESS_KEY"]
    )

    llm = BedrockLLM(
        client=bedrock_client,
        model_id="mistral.mistral-7b-instruct-v0:2"
    )

    embedding_model = BedrockEmbeddings(
        client=bedrock_client,
        model_id="amazon.titan-embed-text-v2:0"
    )

    vectorstore = Chroma(
        collection_name="ai_candidates",
        persist_directory="./chroma_db",
        embedding_function=embedding_model
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chatbot = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

st.set_page_config(page_title="AI Hiring Assistant", page_icon="🧑‍💻", layout="wide")

st.markdown("""
    <style>
        body { background-color: #f5f7fa; }
        .stButton > button {
            width: 100%;
            padding: 10px;
            font-size: 16px;
            border-radius: 8px;
            background-color: #1E90FF;
            color: white;
            border: none;
        }
        .stButton > button:hover {
            background-color: #0073e6;
        }
        .stTextInput > div > div > input {
            border-radius: 8px;
            padding: 8px;
            font-size: 16px;
            border: 1px solid #1E90FF;
        }
        .chat-box {
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            background-color: #eaf2ff;
            box-shadow: 0px 2px 4px rgba(0, 0, 0, 0.1);
        }
        .bot-message {
            background-color: #1E90FF;
            color: white;
            padding: 10px;
            border-radius: 8px;
        }
        .user-message {
            background-color: #333333;
            padding: 10px;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

st.title("AI Hiring Assistant")

col1, col2 = st.columns([1, 2])

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
    st.session_state["candidate_info"] = {}
    st.session_state["tech_stack"] = []
    st.session_state["questions"] = []
    st.session_state["current_question_index"] = 0
    st.session_state["answers"] = []

with col1:
    st.markdown("## 📝 Candidate Details")

    if not st.session_state["candidate_info"]:
        name = st.text_input("Full Name")
        email = st.text_input("Email Address")
        phone = st.text_input("Phone Number")
        experience = st.number_input("Years of Experience", min_value=0, step=1)
        position = st.text_input("Desired Position")

        if st.button("Submit Details"):
            st.session_state["candidate_info"] = {
                "name": name, "email": email, "phone": phone,
                "experience": experience, "position": position
            }
            st.success("Details saved! Now enter your tech stack.")

    else:
        candidate_info = st.session_state["candidate_info"]
        st.success(f"✅ Name: {candidate_info['name']}")
        st.success(f"📧 Email: {candidate_info['email']}")
        st.success(f"📞 Phone: {candidate_info['phone']}")
        st.success(f"💼 Position: {candidate_info['position']}")
        st.success(f"⏳ Experience: {candidate_info['experience']} years")

if st.session_state["candidate_info"] and not st.session_state["tech_stack"]:
    with col1:
        st.markdown("## Enter Your Tech Stack")
        tech_stack_input = st.text_input("Tech Stack (Comma-Separated, e.g., Python, Django, AWS)")

        if st.button("Submit Tech Stack"):
            st.session_state["tech_stack"] = [tech.strip() for tech in tech_stack_input.split(",")]
            st.success("Tech stack saved! Preparing your interview...")

with col2:
    st.markdown("## 🎤 Technical Interview")

    if st.session_state["tech_stack"] and not st.session_state["questions"]:
        tech_stack = st.session_state["tech_stack"]
        experience = st.session_state["candidate_info"].get("experience", 0)

        difficulty = "beginner" if experience < 2 else "intermediate" if experience < 5 else "advanced"

        prompt = (
            f"You are an AI interviewer. Generate 6 to 7 {difficulty}-level technical questions for a candidate "
            f"with expertise in {', '.join(tech_stack)} and {experience} years of experience."
        )

        response = chatbot({"question": prompt})["answer"]
        questions = [q.strip() for q in response.split("\n") if q.strip() and q[0].isdigit()]
        st.session_state["questions"] = questions
        st.session_state["current_question_index"] = 0

    if st.session_state["questions"]:
        current_index = st.session_state["current_question_index"]
        total_questions = len(st.session_state["questions"])

        progress = current_index / total_questions if total_questions else 0
        st.progress(progress)

        if current_index < total_questions:
            current_question = st.session_state["questions"][current_index]
            st.write(f"### Question {current_index+1}/{total_questions}")
            st.info(current_question)

            user_answer = st.text_area("Your Answer", key=f"answer_{current_index}")

            if st.button("Submit Answer", key=f"submit_{current_index}"):
                if user_answer.strip():
                    st.session_state["answers"].append(user_answer)
                    st.session_state["chat_history"].append(("Bot", current_question))
                    st.session_state["chat_history"].append(("User", user_answer))
                    st.session_state["current_question_index"] += 1
                    st.success("Answer submitted! Moving to the next question...")
                    st.rerun()
                else:
                    st.warning("Please provide an answer before submitting.")

if st.button("End Interview"):
    st.success("Thank you for your time!")
    st.session_state.clear()
    st.rerun()
