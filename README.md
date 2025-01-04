# AI-MultiAgent-App

A multi-agent application that explains topics through text (Wikipedia), images (DALL-E), and videos (YouTube).

## Setup Instructions

1. Clone the repository
```bash
git clone <repository-url>
cd AI-MultiAgent-App
```

2. Create and activate virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Unix/MacOS
source .venv/bin/activate
```

3. Install required packages
```bash
pip install langchain-community
pip install langchain-openai
pip install langgraph
pip install langgraph-checkpoint-sqlite
pip install wikipedia
pip install youtube_search
```

4. Environment Configuration
- Create a `.env` file in the root directory
- Add your OpenAI API key
- Configure Claude settings (optional)

5. Test Installation
```bash
python agent.py
```

## Usage Examples

### 1. Wikipedia Text Explanation
Test the Wikipedia tool:
```python
response = execute(agent, query="What is quantum computing?")
```

### 2. DALL-E Image Generation
Test the image generation:
```python
response = execute(agent, query="Show me a visualization of DNA structure")
```

### 3. YouTube Video Search
Test the video search:
```python
response = execute(agent, query="Find a tutorial on Python decorators")
```

## Tool Descriptions

1. **Wikipedia Tool**
   - Used for text-based explanations
   - Provides concise information from Wikipedia
   - Best for: Definitions, concepts, historical facts

2. **DALL-E Tool**
   - Generates custom images
   - High-resolution output (1792x1024)
   - Best for: Visual concepts, illustrations, artistic representations

3. **YouTube Tool**
   - Searches relevant videos
   - Returns video links with descriptions
   - Best for: Tutorials, demonstrations, dynamic concepts

## Example Queries

```python
# Text Example
response = execute(agent, query="Explain what is a Mobius strip")

# Image Example
response = execute(agent, query="Generate an image of a futuristic city")

# Video Example
response = execute(agent, query="Show me how to make sourdough bread")
```

## Configuration

The agent uses a system prompt that determines how to choose between different tools based on the query type. You can modify the system prompt in `agent.py` to customize the behavior.

## Error Handling

The application includes built-in error handling for:
- API failures
- Tool execution errors
- Invalid queries

## Note

Make sure to keep your API keys secure and never commit them to version control.

