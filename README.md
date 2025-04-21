# 🧠 Agentic AI Assistant — LangGraph + Groq

This project implements an intelligent, multi-tool AI assistant using LangGraph and Groq-hosted LLMs. It decomposes compound queries, routes them to specialized tools (calculator, translator, or general assistant), and returns clear, step-by-step responses. Supports full session logging and multi-step reasoning.

---

## 📌 Features

- ✅ Compound query splitting (e.g., "Translate and Multiply")
- ✅ Tool routing with LangGraph
- ✅ Calculator tool (math via LLM)
- ✅ Translator tool (English ↔ German only)
- ✅ General assistant (step-by-step explanations)
- ✅ Groq LLM fallback + model logging
- ✅ JSON session logging per query

---

## 📂 Project Structure


├── chatbot.py                    
├── tools/
│   ├── calculator_tool.py         
│   ├── translation_tool.py       
│   ├── general_tool.py           
│   └── utils.py                  
├── nodes/
│   ├── splitter.py               
│   └── output_node.py             
├── integration_logs/             
├── .env                          
└── README.md



## 🚀 How to Run

1. **Install dependencies:**
   pip install -r requirements.txt

2. **Set your Groq API Key:**
   In a `.env` file:
   GROQ_API_KEY=your_groq_api_key

3. **Start the assistant:**
   python chatbot.py

4. **Try multi-step queries like:**
   - Translate "Good Morning" into German and multiply 5 and 6
   - Tell me the capital of Italy and add 12 and 8

5. **Exit with:**
   exit

---

## 🧠 Sample Session

Ask me anything: Translate "Hello" into German and add 5 and 10

🧠 Final Answer:
[translator] → Hallo
[calculator] → 15

---

## 🗂️ Logs

Each session saves a `session_log_<timestamp>.json` inside `integration_logs/`, containing:

{
  "user_query": "Translate 'Hello' and add 5 and 10",
  "subtasks": [
    {
      "tool": "translator",
      "query": "Translate 'Hello'",
      "response": "Hallo",
      "model_used": "llama3-8b-8192"
    },
    {
      "tool": "calculator",
      "query": "add 5 and 10",
      "response": "15",
      "model_used": "llama3-8b-8192"
    }
  ]
}

---

## ✅ Requirements Completed

- LLM-only general assistant (step-by-step)
- Calculator tool for math
- Translator (English ↔ German)
- Multi-step decomposition + routing
- LangGraph agent structure
- Groq fallback + model tracking
- Full session logging

---

## 👨‍💻 Author

Sriyesh Karampuri  
University of Maryland – MSIS 2025  
sriyeshk@umd.edu
