# 🤖 Chat with Datacrumbs

A Streamlit-based chatbot that lets you interact with the [Datacrumbs.org](https://datacrumbs.org) website using **natural language**.

Built with:
- 🦜 [LangChain](https://www.langchain.com/)
- 🌐 [Google Gemini API](https://ai.google.dev/)
- ⚡ Streamlit for the frontend
- 📄 FAISS for vector search

---

## 🚀 Features

- Chat with the Datacrumbs website content
- Asks questions like:
  - “What bootcamps does Datacrumbs offer?”
  - “Who can contribute to Datacrumbs?”
- Also supports creative queries (e.g., *“If Datacrumbs were an ice cream flavor...”*)
- Built-in question examples
- Dark mode UI with auto suggestions

---

## 🛠️ Tech Stack

| Tech | Description |
|------|-------------|
| `LangChain` | For document loading, chunking, and QA pipeline |
| `Gemini API` | Google’s GenAI model for answering |
| `FAISS` | Local vector search database |
| `Streamlit` | Interactive web app UI |

---

## 🧪 Run It Locally

```bash
# 1. Clone the repo
git clone https://github.com/your-username/chat-with-datacrumbs.git
cd chat-with-datacrumbs

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Create a .env file
echo "GEMINI_API_KEY=your_key_here" > .env

# 5. Run the app
streamlit run app.py
