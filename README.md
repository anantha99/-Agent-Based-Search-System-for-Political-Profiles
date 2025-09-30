# Agent-Based Search System for Political Profiles

An intelligent agent-based system designed to search, research, and generate structured profiles for Indian politicians using Google's Agent Development Kit (ADK) and Gemini AI models.

## Features

- **Entity Disambiguation**: Accurately identifies the correct politician from search queries
- **Multi-Source Research**: Parallel research from government portals, encyclopedias, and recent updates
- **Fact Consolidation**: Merges information from multiple sources into consistent fact sheets
- **Structured Output**: Generates JSON-formatted profiles with Title, Biography, and Current Status
- **Validation**: Ensures profile accuracy and currency using authoritative sources

## Architecture

The system uses a sequential pipeline of specialized agents:

1. **Disambiguation Agent**: Identifies the correct person and provides canonical sources
2. **Parallel Research Agents**:
   - Government Sources: Official roles and portfolios
   - Encyclopedia Sources: Education, career timeline, achievements
   - Recent Updates: Role changes in the last 90 days
3. **Consolidation Agent**: Merges research into a single fact sheet
4. **Extraction Agent**: Produces structured JSON output
5. **Validation Agent**: Verifies accuracy and resolves inconsistencies

## Prerequisites

- Python 3.8+
- Google Cloud Project with Vertex AI enabled
- API keys for Google Search (if using custom search)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/anantha99/-Agent-Based-Search-System-for-Political-Profiles.git
   cd -Agent-Based-Search-System-for-Political-Profiles
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables in `political_profiles_agent/.env`:
   ```
   GOOGLE_API_KEY=your_google_api_key
   GOOGLE_CSE_ID=your_custom_search_engine_id
   ```

## Usage

The system is built using Google's ADK agents. To use the political profile pipeline:

```python
from political_profiles_agent.agent import root_agent

# Run the pipeline for a politician
result = root_agent.run("Give me a profile of Nirmala Sitharam")
print(result["final_profile"])
```

## Output Format

The system produces a structured JSON profile:

```json
{
  "Title": "Minister of Finance and Corporate Affairs",
  "Biography": "Nirmala Sitharaman, born on August 18, 1959, is an Indian economist and politician. She earned her Bachelor's degree in Economics from Seethalakshmi Ramaswami College and a Master's and M.Phil. from Jawaharlal Nehru University. Before entering politics, she worked for organizations like PricewaterhouseCoopers and the BBC World Service in the UK. Her political career began as a member of the National Commission for Women from 2003 to 2005. She joined the Bharatiya Janata Party (BJP) in 2006 and served as its national spokesperson from 2010 to 2014. In 2014, she was inducted into the Union Cabinet and elected to the Rajya Sabha. From 2017 to 2019, Sitharaman made history as India's first full-time female Defence Minister. In May 2019, she was appointed as the Union Minister of Finance and Corporate Affairs, becoming the first full-time woman to hold this office. As Finance Minister, she has presented eight consecutive Union Budgets and has overseen major economic reforms, including the amalgamation of public sector banks and the launch of the Atmanirbhar Bharat initiative. She has been consistently featured in Forbes' list of the world's most powerful women.",
  "Current Status": "Nirmala Sitharaman is currently serving as the Union Minister of Finance and Minister of Corporate Affairs for the Government of India, having been re-appointed in June 2024. She is also a sitting Member of Parliament in the Rajya Sabha, representing the state of Karnataka."
}
```

## Dependencies

Key libraries used:
- `google-adk`: Agent Development Kit for orchestration
- `google-genai`: Gemini AI models
- `pydantic`: Data validation and serialization
- `python-dotenv`: Environment variable management

