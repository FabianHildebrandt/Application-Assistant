---
title: Application Assistant
emoji: 🦛
sdk: gradio
sdk_version: 5.33.0
app_file: app.py
fullWidth: true
header: default
short_description: AI-powered assistant to help you land your dream job.
tags:
- agent-demo-track
pinned: true
---

# 🏆 Application Assistant

### Submission for the: 🤖 Gradio Agents & MCP Hackathon 2025 🚀  
- **Track:** 3 – *Agentic Demo (agent-demo-track)*  
- **Author:** Fabian Hildebrandt

---

## 📌 Overview

**Application Assistant** 🦛🤗 is an intelligent, multi-agent application designed to support users throughout their job application journey.

![Application Assistant](application_assistant.png)

This project demonstrates the power of **multi-agent collaboration** powered by **Gradio**’s UI framework and the **LangGraph**'s agent framework. 

Whether you're writing a compelling cover letter, preparing for an interview, or polishing your CV, **Application Assistant** is here to help.


[Video Tutorial](https://github.com/FabianHildebrandt/Application-Assistant/tree/main) (available on the GitHub Repo)

https://github.com/user-attachments/assets/4f17e8f7-8993-40bb-9de1-ce72c0cab473



## 🤖 What Can It Do?

- 📝 **Cover Letter Generator:** Draft, refine, and customize cover letters tailored to specific job roles.
- 🎤 **Interview Prep:** Simulate mock interview question sets and feedback.
- 👥 **Multi-Agent Collaboration:** Breaks down tasks into sub-tasks handled by specialized agents (e.g., Writer agent, critic agent, interview agent).


## 🧠 Multi-Agent System

The Application Assistant uses a modular agent system:

![Architecture](mas_architecture.png)


These agents communicate and collaborate via a central orchestrator.


## 🚀 Getting Started

### 🔧 Prerequisites
- Python installed
- `pip` installed
- Google AI / NEXUS API key -> [Get a free API key](https://ai.google.dev/gemini-api/docs/api-key)

### 🔐 API Key Configuration

To enable communication with large language models (LLMs), this application requires an API key. You can use either the Google GenAI API or the Nebius API. Feel free to fork this repository to integrate additional [chat models](https://python.langchain.com/docs/integrations/chat/) as needed.

If you choose to use the Google GenAI API, you must set the following environment variables:
	•	GOOGLE_API_KEY: Your Google API key (required)
	•	GOOGLE_DEPLOYMENT_NAME: The specific model deployment name (optional; defaults to gemini-2.0-flash)

You can find the list of available models here: [Gemini models](https://ai.google.dev/gemini-api/docs/models)
You can set environment variables in your shell or define them in a .env file. Example for Unix-based systems:
```
export GOOGLE_API_KEY="your-api-key-here"
export GOOGLE_DEPLOYMENT_NAME="gemini-2.0-flash"
```
For the NEBIUS API, you need to set the environment variable `NEBIUS_KEY`.
If you're using a .env file, make sure to load the file in your application using a package like python-dotenv (load_dotenv()).

### 📥 Installation

```bash
git clone 
cd application-assistant
pip install -r requirements.txt
python app.py
```

---
## 🧭 Using the Tool

Once you've launched the app (by running `python app.py`), open your default browser and navigate to the Gradio interface at `http://localhost:7860`.

![Interface](interface.png)

From there, you can:
1. **Input Your Details** such as job descriptions, existing CV and previous motivation letters and your general motivation.
2. **Interact with the Agents** using the Q&A Chatbot.
3. **Copy** your polished text or copy it directly for use in job applications.

The interface is designed to be intuitive, responsive, and ready to support you at any stage of your job application journey.

The app is also compatible with a Jupyter environment.
