# Define a function to get the color based on sentiment
import matplotlib.pyplot as plt
import textwrap
import re
import spacy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from datetime import datetime
import textwrap
def get_sentiment_color(sentiment):
    sentiment = sentiment.lower() if isinstance(sentiment, str) else ""
    if sentiment == "positive":
        return "green"
    elif sentiment == "negative":
        return "red"
    else:  # neutral or any other value
        return "gray"
    
def format_timeline(timeline_data):
    """
    Format timeline data into properly formatted bullet points.
    
    Parameters:
    -----------
    timeline_data : list or str
        Timeline data either as a list of events or a string with events separated by " - "
        
    Returns:
    --------
    str
        Formatted timeline with bullet points and proper line breaks
    """
    if isinstance(timeline_data, list):
        # If already a list, format each item as a bullet point
        formatted_timeline = "\n\n".join([f"• {event}" for event in timeline_data])
    elif isinstance(timeline_data, str):
        # If it's a string with " - " separators, split and format as bullet points
        events = timeline_data.split(" - ")
        formatted_timeline = "\n\n".join([f"• {event.strip()}" for event in events if event.strip()])
    else:
        formatted_timeline = "No timeline available"
    
    return formatted_timeline

# def format_pattern(timeline_data):
#     """
#     Format timeline data into properly formatted bullet points.
        
#     Parameters:
#     -----------
#     timeline_data : list or str
#         Timeline data either as a list of events or a string with events separated by " - "
            
#     Returns:
#     --------
#     str
#         Formatted timeline with bullet points and proper line breaks
#     """
#     if isinstance(timeline_data, list):
#         # If already a list, format each item as a bullet point
#         formatted_timeline = "\n\n".join([f"• {event}" for event in timeline_data])
#     elif isinstance(timeline_data, str):
#         # If it's a string with " - " separators, split and format as bullet points
#         events = timeline_data.split(".")
#         formatted_timeline = "\n\n".join([f"• {event.strip()}" for event in events if event.strip()])
#     else:
#         formatted_timeline = "No timeline available"
        
#     return formatted_timeline
import re
import spacy 
nlp = spacy.load("en_core_web_sm")

def highlight_complaints(paragraph: str, keyword_file: str) -> str:
    """
    Highlights complaint-related keywords in a paragraph with bold red text using HTML,
    with case-insensitive and lemmatized matching. Returns sentences in bulleted format.

    Parameters:
    - paragraph: The input text to search and format.
    - keyword_file: Path to the text file containing complaint keywords/phrases (one per line).

    Returns:
    - A string with HTML-formatted sentences in bulleted list.
    """
    try:
        with open(keyword_file, "r") as file:
            raw_keywords = [line.strip() for line in file.readlines() if line.strip()]

        # Lemmatize keywords
        lemmatized_keywords = set()
        for phrase in raw_keywords:
            doc = nlp(phrase.lower())
            lemmatized_keywords.add(" ".join([token.lemma_ for token in doc]))

        # Process entire paragraph into sentences
        paragraph_doc = nlp(paragraph)
        bullet_points = []

        for sent in paragraph_doc.sents:
            tokens = [token.text for token in sent]
            lemmas = [token.lemma_.lower() for token in sent]

            result = ""
            i = 0
            while i < len(tokens):
                matched = False
                for phrase in lemmatized_keywords:
                    phrase_lemmas = phrase.split()
                    phrase_len = len(phrase_lemmas)
                    if lemmas[i:i+phrase_len] == phrase_lemmas:
                        original_text = " ".join(tokens[i:i+phrase_len])
                        result += f"<b><span style='color:red'>{original_text}</span></b> "
                        i += phrase_len
                        matched = True
                        break
                if not matched:
                    result += tokens[i] + sent[i].whitespace_
                    i += 1

            bullet_points.append(f"<li>{result.strip()}</li>")

        return f"<ul>\n{''.join(bullet_points)}\n</ul>"

    except FileNotFoundError:
        print(f"Keyword file '{keyword_file}' not found.")
        return paragraph
    
    
def format_pattern(pattern_data):
    """
    Format timeline data into properly formatted bullet points.
    """
    if isinstance(pattern_data, list):
        # If already a list, format each item as a bullet point
        formatted_pattern = "\n\n".join([f"• {event}" for event in pattern_data])
    elif isinstance(pattern_data, str):
        # If it's a string with " - " separators, split and format as bullet points
        events = pattern_data.split(".")
        formatted_pattern = "\n\n".join([f"• {event.strip()}" for event in events if event.strip()])
    else:
        formatted_pattern = "No timeline available"
    
    return formatted_pattern

def format_timeline(timeline_data):
    if isinstance(timeline_data, list):
        # If already a list, format each item as a bullet point
        formatted_timeline = "\n\n".join([f"• {event}" for event in timeline_data])
    elif isinstance(timeline_data, str):
        # If it's a string with " - " separators, split and format as bullet points
        events = timeline_data.split(" - ")
        formatted_timeline = "\n\n".join([f"• {event.strip()}" for event in events if event.strip()])
    else:
        formatted_timeline = "No timeline available"
    return formatted_timeline

def parse_timeline_data(timeline_data):
    """Parse timeline entries to extract dates and descriptions"""
    parsed_data = []
    
    if isinstance(timeline_data, str):
        # Split by bullet points if it's already formatted
        if timeline_data.strip().startswith("•"):
            items = [item.strip()[2:].strip() for item in timeline_data.split("\n\n") if item.strip()]
        else:
            # Split by " - " if it's a string with separators
            items = [item.strip() for item in timeline_data.split(" - ") if item.strip()]
    elif isinstance(timeline_data, list):
        items = timeline_data
    else:
        return []
    
    for item in items:
        # Extract date using regex
        date_match = re.match(r'^([A-Za-z]+ \d+(?:st|nd|rd|th)?, \d{4}):', item)
        if date_match:
            date_str = date_match.group(1)
            description = item[len(date_match.group(0)):].strip()
            
            # Parse the date
            try:
                date_obj = datetime.strptime(date_str, "%B %dst, %Y")
            except ValueError:
                try:
                    date_obj = datetime.strptime(date_str, "%B %dnd, %Y")
                except ValueError:
                    try:
                        date_obj = datetime.strptime(date_str, "%B %drd, %Y")
                    except ValueError:
                        try:
                            date_obj = datetime.strptime(date_str, "%B %dth, %Y")
                        except ValueError:
                            # If all specific formats fail, try a more generic approach
                            cleaned_date = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)
                            date_obj = datetime.strptime(cleaned_date, "%B %d, %Y")
            
            parsed_data.append({
                'date': date_obj,
                'date_str': date_obj.strftime('%b %d, %Y'),
                'description': description
            })
    
    # Sort by date
    parsed_data.sort(key=lambda x: x['date'])
    return parsed_data

def create_matplotlib_vertical_timeline(timeline_data, figsize=(9, 9), save_path=None):
    """Create a vertical timeline visualization using Matplotlib for dark background use (e.g., in Streamlit dark theme)"""
        
    parsed_data = parse_timeline_data(timeline_data)
        
    if not parsed_data:
        print("No valid timeline data to plot")
        return None

    # Create figure and axis with dark background
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#0e1117')  # Streamlit dark theme background
    ax.set_facecolor('#0e1117')         # Match axis background

    # Calculate positions
    num_events = len(parsed_data)
    positions = list(range(num_events, 0, -1))  # Reverse for top-down

    # Extract timeline elements
    dates = [item['date_str'] for item in parsed_data]
    descriptions = [item['description'] for item in parsed_data]
    
    ax.plot([0.1] * num_events, positions, 'o-', color='gray', markersize=12, markerfacecolor='#3498db')
    
    # Add dates on the left side
    for i, (date, pos) in enumerate(zip(dates, positions)):
        ax.text(0.05, pos, date, ha='right', va='center', fontsize=12,color='white')
    
    # Add descriptions on the right side
    for i, (desc, pos) in enumerate(zip(descriptions, positions)):
        # Wrap text for better display
        wrapped_text = textwrap.fill(desc, width=50)
        ax.text(0.15, pos, wrapped_text, ha='left', va='center', fontsize=11,color='white')

    # # Draw vertical line
    # ax.plot([0.5] * num_events, positions, 'o-', color='white', markersize=12, markerfacecolor='#1f77b4')
    
    # # Add dates (left side)
    # for i, (date, pos) in enumerate(zip(dates, positions)):
    #     ax.text(0.45, pos, date, ha='right', va='center', fontsize=10, color='white')

    # # Add descriptions (right side)
    # for i, (desc, pos) in enumerate(zip(descriptions, positions)):
    #     wrapped_text = textwrap.fill(desc, width=50)
    #     ax.text(0.55, pos, wrapped_text, ha='left', va='center', fontsize=9, color='white')

    # Title
    #ax.set_title('Event Timeline', fontsize=14, fontweight='bold', color='white')

    # Hide spines and ticks
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)

    # Set limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, num_events + 1)

    plt.tight_layout()

    # if save_path:
    #     plt.savefig(save_path, bbox_inches='tight', dpi=300, facecolor=fig.get_facecolor())

    return fig, ax