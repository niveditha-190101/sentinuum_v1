from typing import TypedDict, Optional
import json
from schema import schema
from prompts import tagging_prompt, extraction_prompt
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI()
class State(TypedDict):
    conversation: str
    tagged_output: Optional[str]
    summary: Optional[dict]

# Conversation Tagger Agent
class ConversationTagger:
    def __init__(self, model="gpt-4o-mini"):
        self.model = model
    
    def tag(self, state: State) -> State:
        client = OpenAI()
        prompt = f"""{tagging_prompt}
            Transcript:
            {state['conversation']}
            """
        
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": prompt},
            ]
        )
        tagged = response.choices[0].message.content
        
        return {**state, "tagged_output": tagged}

# Insights Extractor Agent
class InsightsExtractor:
    def __init__(self, schema, model="gpt-4o"):
        self.schema = schema
        self.model = model
    
    def extract(self, state: State) -> State:
        client = OpenAI()
        prompt = f"""{extraction_prompt}
        Conversation: {state['tagged_output']}
        """
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            tools=[{"type": "function", "function": {"name": "conversation_summary", "parameters": self.schema}}],
            tool_choice="auto",
            temperature=0.15,
            seed=1
        )
        summary = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        return {**state, "summary": summary}

# Function to process a conversation
def analyze_conversation(conversation_text):
    tagger = ConversationTagger()
    summarizer = InsightsExtractor(schema)
    
    # Process through pipeline
    tagged_state = tagger.tag({"conversation": conversation_text})
    output = summarizer.extract(tagged_state)
    
    return output