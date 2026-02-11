# FBI Case Story Generator

An agentic, multi-step AI system that autonomously selects a historical FBI case, converts it into a narrative story, and generates images and narrated audio, all by using open-source models

---

## Project Overview

The system performs the following steps autonomously:

1. Scrapes the FBI “Famous Cases” archive
2. Selects a compelling historical case
3. Extracts and cleans factual source material
4. Converts the facts into a long-form narrative story
5. Splits the story into structured sections
6. Generates:
   - An image for each section
   - Text-to-speech narration
   - A combined narrated audio file
7. Outputs structured JSON containing all generated artifacts

All models used are open-source and run locally (or in Google Colab).

---

## Architecture Evolution

This project has gone through multiple architectural changes:

### v1 — smolAgents (Old and Dysfunctional)
- Initial exploration of tool-based agents
- Focused on understanding agent planning and tool execution
- Retained in the repository for reference and experimentation
- However, very buggy and not suited for task at hand

### v2 — LangGraph (Current Direction)
- Rewritten using explicit graph-based state transitions
- Improved reliability, debuggability, and control flow
- Clear separation of concerns between nodes (scraping, writing, asset generation)
- Deterministic execution with structured state passing

---

## Technologies Used

- Python
- LangGraph (graph-based agent orchestration)
- LangChain Core
- Transformers (local LLM inference)
- Diffusers (Stable Diffusion image generation)
- Torch / Torchaudio
- BeautifulSoup + Requests (web scraping)
- Pydub + SoundFile (audio processing)


