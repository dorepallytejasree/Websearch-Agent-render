import os
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# LangChain imports
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini LLM

# Google Gemini SDK
import google.generativeai as genai

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("❌ GEMINI_API_KEY is missing in .env file!")

print(f"🔑 Loaded GEMINI_API_KEY: {GEMINI_API_KEY[:4]}****{GEMINI_API_KEY[-4:]}")

# Configure Gemini API
genai.configure(api_key=GEMINI_API_KEY)

print("\n🔍 Available Gemini models supporting generateContent:")
try:
    models = genai.list_models()
    for m in models:
        if "generateContent" in getattr(m, "supported_generation_methods", []):
            print(" -", m.name)
except Exception as e:
    print("⚠️ Could not list models:", e)
print("=================================\n")

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)  # optionally add origins=['http://localhost:3000']

ddg = DuckDuckGoSearchAPIWrapper()
duckduckgo_tool = Tool(
    name="DuckDuckGoSearch",
    func=lambda q: ddg.run(q),
    description="Use this to search the web for up-to-date information."
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  
    temperature=0,
    google_api_key=GEMINI_API_KEY
)

agent = initialize_agent(
    tools=[duckduckgo_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "⚠️ No question provided"}), 400

    try:
        answer = agent.run(question)
        return jsonify({"answer": str(answer)})
    except Exception as e:
        error_trace = traceback.format_exc()
        print("❌ ERROR:", error_trace)
        return jsonify({
            "error": "Sorry, something went wrong while processing your request.",
            "details": str(e)
        }), 500

@app.route("/health")
def health():
    return jsonify({"status": "ok", "message": "Server is running."})

@app.errorhandler(404)
def page_not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
