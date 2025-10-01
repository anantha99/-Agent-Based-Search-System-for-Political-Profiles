# Agent-Based Search System for Political Profiles

An intelligent CLI application that uses Google's Agent Development Kit (ADK) and Gemini AI models to research and generate structured profiles for Indian politicians. The system features intelligent routing, multi-source research, and comprehensive validation to ensure accurate and up-to-date political profiles.

## Features

- **Smart Entity Recognition**: Automatically identifies whether a query refers to an Indian politician
- **Intelligent Routing**: Provides helpful responses for non-politicians instead of forcing profile generation
- **Multi-Source Research**: Parallel research from government portals, encyclopedias, and recent updates
- **Fact Consolidation**: Merges information from multiple sources into consistent, citation-rich fact sheets
- **Structured Output**: Generates validated JSON-formatted profiles with Title, Biography, and Current Status
- **Beautiful CLI Interface**: Rich terminal output with progress indicators and formatted results
- **Comprehensive Validation**: Ensures profile accuracy using authoritative sources

## Architecture

The system uses a agent-based architecture with intelligent routing:

### Core Components

1. **Router Agent**: Intelligently determines if the query is about an Indian politician
   - If politician: Proceeds to research pipeline
   - If not politician: Provides friendly guidance message

2. **Research Pipeline** (for politicians only):
   - **Disambiguation Agent**: Validates and normalizes the politician's identity
   - **Parallel Research Agents** (run simultaneously):
     - **Government Sources**: Official roles, ministries, and parliamentary positions
     - **Encyclopedia Sources**: Education, career timeline, and achievements
     - **Recent Updates**: Role changes and updates in the last 90 days
   - **Consolidation Agent**: Merges research into a single, consistent fact sheet
   - **Extraction Agent**: Produces structured JSON profile
   - **Validation Agent**: Verifies accuracy and resolves any inconsistencies

## Prerequisites

- Python 3.8+
- Google Cloud Project with Vertex AI enabled (optional)
- Google API key with Gemini AI access

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/anantha99/-Agent-Based-Search-System-for-Political-Profiles.git
   cd -Agent-Based-Search-System-for-Political-Profiles
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables in `political_profiles_agent/.env`:
   ```bash
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Usage

### Command Line Interface (Recommended)

The system provides a beautiful CLI interface with real-time progress indicators:

```bash
# Interactive mode (prompts for name)
python main.py

# Direct mode (specify name as argument)
python main.py --name "Nirmala Sitharaman"

# Short form
python main.py -n "Narendra Modi"
```
### Google ADK WEB UI

```bash
# Web Ui hosted locally open the local and enter the user query

adk web
```
