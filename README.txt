
# DCPD ChatBot: A Q&A Application for Dubai Club for People of Determination

## Overview
The **DCPD ChatBot** is a Streamlit-based application designed to answer user questions about tournaments, results, and general information related to the Dubai Club for People of Determination. It leverages LangChain, OpenAI APIs, and FAISS for document-based conversational AI.

---

## Features
1. **User-Friendly Chat Interface**: 
   - Ask questions about tournaments, results, and schedules.
   - Clear and precise responses based on provided documents.

2. **Document Integration**:
   - Supports PDF and TXT file formats.
   - Extracts and processes data for accurate Q&A.

3. **Vector Database for Fast Retrieval**:
   - FAISS (Facebook AI Similarity Search) is used for efficient similarity search.
   - Embedding storage for fast response times.

4. **Customizable Prompt Template**:
   - Ensures responses are based solely on provided context.
   - If information is unavailable, the bot clearly states it.

---

## How It Works
1. **Document Loading**:
   - Upload PDF and TXT files containing relevant information.
   - Text is extracted and preprocessed into manageable chunks.

2. **Embedding Creation**:
   - Text chunks are embedded using OpenAI's embeddings.
   - Data is stored in a FAISS vector database for similarity search.

3. **User Interaction**:
   - Users input questions through the chat interface.
   - The chatbot searches the vector database for relevant chunks.
   - Responses are generated using OpenAI's `gpt-3.5-turbo`.

4. **Context-Driven Responses**:
   - Responses are derived from the context within the provided documents.
   - If the answer is not found, the bot informs the user.

---

## Prerequisites
1. **Python 3.8+**
2. **Required Libraries**:
   - `streamlit`
   - `langchain`
   - `langchain-openai`
   - `faiss-cpu`
   - `PyMuPDF`
   - `python-dotenv`
3. **Environment Variables**:
   - OpenAI API Key: Set in a `.env` file under the variable `OPENAI_API_KEY`.

---

## Installation and Setup

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/DCPD-ChatBot.git
   cd DCPD-ChatBot
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Add Environment Variables**:
   - Create a `.env` file:
     ```bash
     touch .env
     ```
   - Add your OpenAI API Key:
     ```
     OPENAI_API_KEY=your_openai_api_key
     ```

4. **Prepare Documents**:
   - Place your PDF and TXT files in the `documents/` directory.

5. **Run the Application**:
   ```bash
   streamlit run app.py
   ```

---

## Usage
1. Open the app in your browser (Streamlit will provide the URL).
2. Ask questions in the chat box.
3. View detailed responses based on the uploaded documents.

---

## Folder Structure
```
DCPD-ChatBot/
│
├── app.py                 # Main application code
├── requirements.txt       # List of dependencies
├── documents/             # Directory for PDFs and TXT files
├── logo.jpg               # Club logo for the app interface
├── .env                   # Environment variables file (not tracked in Git)
└── README.md              # Project documentation
```

---

## Example Questions
- "What tournaments are scheduled?"
- "Who won the last basketball game?"
- "Can you provide information about the club’s facilities?"

---

## Customization
You can modify:
- **Prompt Template**: Located in `get_conversational_chain()`.
- **Document Paths**: Update the `pdf_docs` and `txt_files` variables in the `main()` function.

---

## Troubleshooting
- **Error Processing Files**: Ensure files are correctly placed in the `documents/` folder.
- **No Response**: Check your `.env` file for the correct OpenAI API key.

---

## License
This project is open-source. Feel free to customize and extend its functionality to meet your needs.
