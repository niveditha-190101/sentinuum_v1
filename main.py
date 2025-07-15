## Uses Langgraph for workflow management and analysis
## The older version of the code 
## env is agnetic_rag 

from typing import List, Dict, Any, Optional, Union, Tuple, Sequence, Annotated, TypedDict
import os
import json
from langgraph.graph import StateGraph, END
import streamlit as st
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import Tool, tool
from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from datetime import datetime
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv
from visuals import get_sentiment_color, format_timeline,highlight_complaints,format_pattern,create_matplotlib_vertical_timeline
from conversation_analyser import analyze_conversation
from langchain.text_splitter import RecursiveJsonSplitter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load environment variables from .env file
load_dotenv()


class WorkflowState(TypedDict):
    chat_history: List[HumanMessage]
    transcript_content: Optional[str]
    analysis_result: Optional[str]
    root_cause_analysis: Optional[Dict]
    confidence_score: Optional[float]
    final_summary: Optional[str]
    timestamp: Optional[str]

def process_transcript_content(transcript_content):
    """
    Process transcript content directly instead of reading from a file.
    
    Args:
        transcript_content: The actual transcript text
        
    Returns:
        JSON string containing the content
    """
    try:
        print(f"\n=== Process Transcript Debug ===")
        print(f"Processing transcript content")
        
        if not transcript_content:
            error_result = json.dumps({"error": "Transcript content is empty"})
            print(f"Output: {error_result}")
            return error_result
        
        # Ensure content is properly escaped and formatted
        result = json.dumps({
            "content": transcript_content,
            "source": "direct_input"
        }, ensure_ascii=False)
        
        print(f"Successfully processed transcript")
        print(f"Content length: {len(transcript_content)}")
        print(f"Output JSON format: {result[:200]}...")  # Print first 200 chars
        return result
        
    except Exception as e:
        error_result = json.dumps({"error": f"Error processing transcript: {str(e)}"})
        print(f"Error: {str(e)}")
        print(f"Output: {error_result}")
        return error_result
    finally:
        print("=== End Process Transcript Debug ===\n")

# Function to handle tagged_output
def handle_tagged_output(tagged_output):
    return process_transcript_content(tagged_output.get("tagged_output"))

def analyze_content(content_json: str) -> str:
    """
    Analyze transcript content for summary, issue, and status.
    
    Args:
        content_json: JSON string containing transcript content
        
    Returns:
        JSON string containing analysis results
    """
    try:
        llm = ChatOpenAI(model="gpt-4o-mini")
        print("\n=== Analyze Content Debug ===")
        print(f"Input type: {type(content_json)}")
        
        if not content_json:
            error_result = json.dumps({"error": "Empty input received"})
            print(f"Output: {error_result}")
            return error_result
        
        try:
            content_data = json.loads(content_json)
            print("Successfully parsed JSON input")
        except json.JSONDecodeError as e:
            print(f"Initial JSON parse error: {str(e)}")
            print(f"Problematic input: {content_json[:200]}...")
            content_data = {"content": content_json}
            print("Handled input as raw text")
        
        content = content_data.get("content", "")
        if not content:
            error_result = json.dumps({"error": "No content found in input"})
            print(f"Output: {error_result}")
            return error_result
        
        print(f"Content length for analysis: {len(content)}")
        print("Starting LLM analysis...")
        
        # Summary analysis
        summary_prompt = PromptTemplate(
            template="Summarize the following text:\n\n{content}\n\nSummary:",
            input_variables=["content"]
        )
        summary = llm.invoke(summary_prompt.format(content=content))
        print("Generated summary")
        #
        name_prompt = PromptTemplate(
            template="extract the name of the person whos talking not the agent or customer care:\n\n{content}\n\nName:",
            input_variables=["content"]
        )
        name = llm.invoke(name_prompt.format(content=content))
        print("Name Identified")
        # Issue identification
        issue_prompt = PromptTemplate(
            template="From the following transcript, identify the main issue:\n\n{content}\n\nMain Issue:",
            input_variables=["content"]
        )
        issue = llm.invoke(issue_prompt.format(content=content))
        print("Identified issue")
        
        # Status determination
        status_prompt = PromptTemplate(
            template="From the following transcript, identify the status (resolved or pending):\n\n{content}\n\nStatus:",
            input_variables=["content"]
        )
        status = llm.invoke(status_prompt.format(content=content))
        print("Determined status")
        
        # Compile analysis results
        analysis_result = {
            "summary": str(summary.content),
            "name" : str(name.content),
            "issue": str(issue.content),
            "status": str(status.content),
            "file_path": content_data.get("file_path", "")
        }
        
        result = json.dumps(analysis_result, ensure_ascii=False)
        print(result)
        print("Successfully created analysis result")
        print(f"Output JSON format: {result[:200]}...")
        return result
        
    except Exception as e:
        error_result = json.dumps({"error": f"Error in analysis: {str(e)}"})
        print(f"Error: {str(e)}")
        print(f"Output: {error_result}")
        return error_result
    finally:
        print("=== End Analyze Content Debug ===\n")

def perform_root_cause_analysis(issue_data: str) -> Dict[str, Any]:
    """
    Performs RAG-based root cause analysis using the knowledge base
    """
    # file_path = "D:/vanguard/Sentiment Integrated/collated.json"
    file_path = "C:/Users/niveditha.n.lv/Documents/summarizer/sentinuum v1/collated_v2.json"
    # transaction_data = 'D:/vanguard/Sentiment Integrated/transaction.json'    
    transaction_data = "C:/Users/niveditha.n.lv/Documents/summarizer/sentinuum v1/transaction data.json"    
    
    try:
        # Parse the current issue
        current_data = json.loads(issue_data)
        issue_description = current_data.get('issue', '')
        extracted_name = current_data.get('name','')    
        if not issue_description:
            return {
                "root_cause": "No issue description available",
                "patterns": [],
                "timeline": "No timeline available",
                "rationale": "No analysis possible"
            }
            
        #Setup RAG components
        embedding_function = OpenAIEmbeddings()
        loader = JSONLoader(file_path=file_path, jq_schema=".[]", text_content=False)
        loader_transaction = JSONLoader(file_path=transaction_data, jq_schema=".[]", text_content=False)
        documents = loader.load()
        transaction_documents = loader_transaction.load()
        db = FAISS.from_documents(documents, embedding_function)
        db_transaction = FAISS.from_documents(transaction_documents, embedding_function)    
            
        # Search for underlying patterns and root causes
        retriever = db.as_retriever(
            search_kwargs={"k": 2}  # Get enough context for pattern analysis
        )
        transaction_retriever = db_transaction.as_retriever(
            search_kwargs={"k": 2}  # Transaction Data 
        )
        query = f"{extracted_name} {issue_description}"
        context_docs = retriever.get_relevant_documents(query)
        transaction_doc= transaction_retriever.get_relevant_documents(query)
        
        context = "\n".join([doc.page_content for doc in context_docs])
        transaction_context =  "\n".join([doc.page_content for doc in transaction_doc])   
        template = """
        Knowledge Base Context: ${context}
        Transaction Context: ${transaction_context}

       
        
        Analysis Instructions
        You are tasked with performing a comprehensive root cause analysis by connecting information between the knowledge base context and transaction data. Your goal is to identify patterns, relationships, and causal factors that explain the situation.
        Required Analysis Tasks
        for the name {name}
        and issue {issue}
       
        1. Timeline Integration Analysis

        Extract all timestamps from both the knowledge base context and transaction data
        Create a unified chronological timeline incorporating both data sets
        Identify temporal correlations between customer interactions/issues and financial transactions
        Examine transaction patterns (withdrawals, deposits, liquidations) in relation to key events in {context}
        Complete the timeline in 6 exact sentences include important events in that 

        2. Root Cause Identification
        Explain everything in 5 W's , why is the customer disatisfied, what is the issue, what steps are taken, what is the result 

        3. Pattern Recognition

        Identify recurring patterns or trends across the data
        Note specific triggers that appear to initiate transaction behaviors
        Identify common failure points where customer issues and transaction problems intersect
        Detect unusual transaction patterns and their relation to customer interactions

        REQUIRED OUTPUT FORMAT
        Your analysis must be provided in exactly the following format with these precise section headers:
        ROOT_CAUSE: Explain everything in 5 W's , why is the customer disatisfied, when did this happen, what is the issue, what steps are taken, what is the result?
        PATTERNS: Detail the recurring patterns discovered across both datasets. Include specific triggers, common failure points, and any unusual transaction behaviors (continuous withdrawals/deposits, liquidation patterns, etc.) that correlate with customer interactions. Highlight significant transaction anomalies and their connection to customer issues. Explain all these in 5 sentences short and consise
        TIMELINE: Present a chronological organization of all relevant events from both datasets. Include specific dates from both contexts, showing how they correlate. Highlight important milestones, decision points, initial contacts, escalations, follow-up actions, and resolution attempts. Show clear connections between dates in the knowledge base and transaction activities 6 sentences from ths timeline having the important information.Stick to this format example September 10th, 2023 
        RATIONALE: Explain your reasoning process and why you believe these connections exist between the knowledge base context and transaction data. Include your analysis of how customer behavior correlates with transaction activities and what process failures might be occurring. Provide evidence-based justification for your conclusions and show me the numbers from the transaction data that is related to the analysis.Also provide few actionable recomendations
        Important Notes 

        Each section must start exactly with the section header (ROOT_CAUSE:, PATTERNS:, etc.) followed by your analysis
        Be comprehensive but concise in your analysis
        Ensure you're making explicit connections between the knowledge base context and transaction context
        Focus on specific evidence and avoid general statements
        Include all relevant dates and look for temporal correlations between contexts"""
        # Create and execute analysis chain
        prompt = ChatPromptTemplate.from_template(template)
        model = ChatOpenAI(temperature=0)
        chain = prompt | model | StrOutputParser()
            
        response = chain.invoke({
            "name": extracted_name,
            "issue": issue_description,
            "context": context,
            "transaction_context":transaction_context
        })
            
        # Parse response
        analysis_results = {}
        current_key = None
        current_content = []
            
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('ROOT_CAUSE:'):
                current_key = 'root_cause'
                current_content = [line.split('ROOT_CAUSE:', 1)[1].strip()]
            elif line.startswith('PATTERNS:'):
                if current_key:
                    analysis_results[current_key] = ' '.join(current_content)
                current_key = 'patterns'
                current_content = [line.split('PATTERNS:', 1)[1].strip()]
            elif line.startswith('TIMELINE:'):
                if current_key:
                    analysis_results[current_key] = ' '.join(current_content)
                current_key = 'timeline'
                current_content = [line.split('TIMELINE:', 1)[1].strip()]
            elif line.startswith('RATIONALE:'):
                if current_key:
                    analysis_results[current_key] = ' '.join(current_content)
                current_key = 'rationale'
                current_content = [line.split('RATIONALE:', 1)[1].strip()]
            elif current_key:
                current_content.append(line)
            
        if current_key:
            analysis_results[current_key] = ' '.join(current_content)
            
        return analysis_results
            
    except Exception as e:
        print(f"Error in root cause analysis: {str(e)}")
        return {
            "root_cause": f"Analysis error: {str(e)}",
            "patterns": [],
            "timeline": "Timeline analysis failed",
            "rationale": "Analysis failed"
        }

def create_workflow():
    """Create and configure the workflow graph"""
    workflow = StateGraph(WorkflowState)
        
    # Define processing nodes
    def process_transcript_node(state: WorkflowState):
        # Extract the actual transcript content from the chat history
        transcript_content = state["chat_history"][-1].content
        
        # If the content is formatted as "tag: content", extract just the content part
        if ": " in transcript_content:
            parts = transcript_content.split(": ", 1)
            if len(parts) > 1:
                transcript_content = parts[1]
        
        # Process the transcript content directly
        result = process_transcript_content(transcript_content)
        # result = transcript_content
        return {"transcript_content": result}
        
    def analyze_content_node(state: WorkflowState):
        if not state.get("transcript_content"):
            return {"analysis_result": json.dumps({"error": "No transcript content"})}
        result = analyze_content(state["transcript_content"])
        print(result)
        return {"analysis_result": result}
        
    def root_cause_node(state: WorkflowState):
        if not state.get("analysis_result"):
            return {
                "root_cause_analysis": None,
                "timeline": "No timeline available",
                "error": "Missing analysis result"
            }
            
        analysis_results = perform_root_cause_analysis(state["analysis_result"])
            
        return {
            "root_cause_analysis": analysis_results,
            "timeline": analysis_results.get('timeline', 'No timeline available')
        }
        
    def create_summary_node(state: WorkflowState):
        if not state.get("root_cause_analysis"):
            return {
                "final_summary": "Analysis could not be completed",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
        root_cause_analysis = state["root_cause_analysis"]
            
        summary = {
            "root_cause": root_cause_analysis.get("root_cause", "Not available"),
            "patterns": root_cause_analysis.get("patterns", "Not available"),
            "timeline": root_cause_analysis.get("timeline", "No timeline available"),
            "rationale": root_cause_analysis.get("rationale", "Not available"),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
            
        return {
            "final_summary": json.dumps(summary),
            "timestamp": summary["timestamp"]
        }
        
    def router(state: WorkflowState) -> str:
        if not state.get("transcript_content"):
            return "process_transcript"
        if not state.get("analysis_result"):
            return "analyze_content"
        if not state.get("root_cause_analysis"):
            return "root_cause"
        if not state.get("final_summary"):
            return "create_summary"
        return "end"
        
    # Add nodes to workflow
    workflow.add_node("process_transcript", process_transcript_node)
    workflow.add_node("analyze_content", analyze_content_node)
    workflow.add_node("root_cause", root_cause_node)
    workflow.add_node("create_summary", create_summary_node)
        
    # Set entry point
    workflow.set_entry_point("process_transcript")
        
    # Add conditional edges
    for node in ["process_transcript", "analyze_content", "root_cause", "create_summary"]:
        workflow.add_conditional_edges(
            node,
            router,
            {
                "process_transcript": "process_transcript",
                "analyze_content": "analyze_content",
                "root_cause": "root_cause",
                "create_summary": "create_summary",
                "end": END
            }
        )
        
    return workflow.compile()

def create_analysis_df(results: list) -> pd.DataFrame:
    """Create DataFrame from analysis results"""
    df_data = []
        
    for result in results:
        if isinstance(result.get('final_summary'), str):
            summary_data = json.loads(result['final_summary'])
            df_data.append({
                'Transcript': os.path.basename(result.get('transcript_path', '')),
                'Root_Cause': summary_data.get('root_cause', 'Not available'),
                'Patterns': summary_data.get('patterns', 'Not available'),
                'Timeline': summary_data.get('timeline', 'No timeline available'),
                'Rationale': summary_data.get('rationale', 'Not available'),
                'Timestamp': summary_data.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            })
        
    return pd.DataFrame(df_data)


def process_transcript(transcript_text: str) -> Dict[str, Any]:
    """
    Process a transcript through the LangGraph workflow.
        
    Args:
        transcript_text: The actual transcript text
            
    Returns:
        Dict containing the workflow results
    """
    workflow = create_workflow()
        
    # Initialize state with correct structure
    initial_state = {
        "chat_history": [HumanMessage(content=transcript_text)],
        "transcript_content": None,
        "analysis_result": None,
        "root_cause_analysis": None,
        "timeline": None,
        "final_summary": None,
        "timestamp": None
    }
        
    try:
        # Execute workflow
        result = workflow.invoke(initial_state)
            
        # Parse final summary if available
        if result.get("final_summary"):
            try:
                summary_data = json.loads(result["final_summary"])
                output = {
                    "transcript_preview": transcript_text[:100] + "..." if len(transcript_text) > 100 else transcript_text,
                    "root_cause": summary_data.get("root_cause", "Not available"),
                    "timeline": summary_data.get("timeline", "No timeline available"),
                    "timestamp": result.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    "analysis_complete": True
                }
            except json.JSONDecodeError:
                output = {
                    "transcript_preview": transcript_text[:100] + "..." if len(transcript_text) > 100 else transcript_text,
                    "root_cause": result["final_summary"],
                    "timeline": result.get("timeline", "No timeline available"),
                    "timestamp": result.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    "analysis_complete": True
                }
        else:
            output = {
                "transcript_preview": transcript_text[:100] + "..." if len(transcript_text) > 100 else transcript_text,
                "error": "Workflow did not complete successfully",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "analysis_complete": False
            }
            
        # Add any intermediate analysis if available
        if result.get("analysis_result"):
            try:
                analysis_data = json.loads(result["analysis_result"])
                output["initial_analysis"] = analysis_data
            except json.JSONDecodeError:
                output["initial_analysis"] = result["analysis_result"]
            
        if result.get("root_cause_analysis"):
            output["detailed_analysis"] = result["root_cause_analysis"]
            
        # Update Streamlit session state if available
    
            # Initialize the session state dataframe if not already there
            if 'responses_df' not in st.session_state:
                st.session_state.responses_df = pd.DataFrame(
                    columns=['transcript', 'response', 'timestamp']
                )

            # Create a new row for the latest output
            new_row = pd.DataFrame([{
                'transcript': transcript_text[:100] + "..." if len(transcript_text) > 100 else transcript_text,
                'response': json.dumps(output),
                'timestamp': output.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            }])

            # Append to the session-level dataframe
            st.session_state.responses_df = pd.concat(
                [st.session_state.responses_df, new_row],
                ignore_index=True
            )

            print(output)
            return output

    except Exception as e:
            error_output = {
                "transcript_preview": transcript_text[:100] + "..." if len(transcript_text) > 100 else transcript_text,
                "error": f"Workflow error: {str(e)}",
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "analysis_complete": False
            }

            print(f"Error processing transcript: {str(e)}")
            return error_output

def display_analysis_result(result: Dict):
    """Display analysis result in a clean, organized format"""
    
    # complaints = 'D:/vanguard/Sentiment Integrated/complaints.txt'
    complaints = "C:/Users/niveditha.n.lv/Documents/summarizer/sentinuum v1/complaint_keywords.txt"

    # Root Cause Analysis from detailed_analysis
    if 'detailed_analysis' in result:
        try:
            # Try to parse the detailed_analysis if it's a JSON string
            if isinstance(result['detailed_analysis'], str):
                try:
                    detailed = json.loads(result['detailed_analysis'])
                except json.JSONDecodeError:
                    detailed = {"content": result['detailed_analysis']}
            else:
                detailed = result['detailed_analysis']
                
            st.subheader("Root Cause Analysis")
            if isinstance(detailed, dict):
                st.markdown(detailed.get('root_cause', 'Not available'))
                    
                st.subheader("Patterns")
                pattern = detailed.get('patterns', 'Not available')
                formatted_pattern = format_pattern(pattern)
                st.markdown(formatted_pattern, unsafe_allow_html=True)
                # st.markdown(detailed.get('patterns', 'Not available'))
                    
                # Usage example:
                
                st.subheader("Timeline")
                timeline = detailed.get('timeline', 'No timeline available')
                print(timeline)
                if timeline != 'No timeline available':
                    result_1 = create_matplotlib_vertical_timeline(timeline)
    
                    if result_1:
                        fig, _ = result_1  # Unpack safely (you can ignore ax with `_`)
                        st.pyplot(fig)
                    else:
                        st.write("No valid timeline data to display")
                else:
                    st.write("No timeline available")
                # formatted_timeline = format_timeline(timeline)
                # st.markdown(formatted_timeline, unsafe_allow_html = True)
                    
                st.subheader("Analysis Rationale")
                rationale = detailed.get('rationale', 'Not available')
                highlighted_text = highlight_complaints(rationale, complaints)
                st.markdown(highlighted_text, unsafe_allow_html=True)
                
            else:
                st.markdown(str(detailed))
        except Exception as e:
            st.error(f"Error displaying detailed analysis: {str(e)}")
            
            st.markdown(str(result['detailed_analysis']))
        
    # Initial Analysis summary
    if 'initial_analysis' in result:
        with st.expander("View Initial Analysis", expanded=False):
            try:
                # Try to parse the initial_analysis if it's a JSON string
                if isinstance(result['initial_analysis'], str):
                    try:
                        initial = json.loads(result['initial_analysis'])
                    except json.JSONDecodeError:
                        initial = {"content": result['initial_analysis']}
                else:
                    initial = result['initial_analysis']
                    
                if isinstance(initial, dict):
                    st.markdown("### Summary")
                    st.markdown(initial.get('summary', 'Not available'))
                        
                    st.markdown("### Issue")
                    st.markdown(initial.get('issue', 'Not available'))
                        
                    st.markdown("### Status")
                    st.markdown(initial.get('status', 'Not available'))
                else:
                    st.markdown(str(initial))
            except Exception as e:
                st.error(f"Error displaying initial analysis: {str(e)}")
                st.markdown(str(result['initial_analysis']))

def display_transcript_analysis():     
    # st.title("Sentinuum AI ")
        
    # # Initialize session state     
    # if 'processed_results' not in st.session_state:         
    #     st.session_state.processed_results = []     
    # if 'responses_df' not in st.session_state:         
    #     st.session_state.responses_df = pd.DataFrame(
    #         columns=['transcript', 'response', 'timestamp']
    #     )
        
    # # Text area for pasting conversation     
    # conversation_input = st.text_area(
    #     "Paste your unprocessed conversation transcript here:",
    #     height=300,
    #     placeholder="Hi, I have an issue with my account.\nHello, how can I help you today?..."
    # )
        
    # # Process button     
    # if st.button("Process Conversation", type="primary"):         
    #     if not conversation_input:             
    #         st.error("Please paste a conversation transcript before processing.")             
    #         return
                
    #     with st.status("Processing conversation...") as status:             
    #         st.write("Analyzing conversation...")
                    
    #         # Process the conversation text directly             
    #         tagged_output = analyze_conversation(conversation_input)
                    
    #         # Process the tagged transcript             
    #         result = process_transcript(tagged_output.get('tagged_output'))
                    
    #         # Add to results with a timestamp
    #         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #         result_with_metadata = {
    #             'transcript': conversation_input[:100] + "..." if len(conversation_input) > 100 else conversation_input,
    #             'processed_result': result,
    #             'tagged_output': tagged_output,
    #             'timestamp': timestamp
    #         }
                    
    #         st.session_state.processed_results.append(result_with_metadata)
                
    #         # Display processing complete message
    #         status.update(label="✅ Processing complete!", state="complete")
                
    #     # Create tabs for different views
    #     tab1, tab2 = st.tabs(["Transcript Breakdown" , "Analysis Results"])
            
    #     # Tab 2: Analysis Results
    #     with tab2:
    #         display_analysis_result(result)
            
    #     # Tab 1: Tagged Transcript
    #     with tab1:
    #         display_beautified_transcript(tagged_output.get('summary'))


    #         # st.subheader("Tagged Transcript")
    #         # # Estimate required height based on content
    #         # num_lines = tagged_output.get('tagged_output').count('\n') + 1
    #         # display_height = max(500, min(num_lines * 20, 1000))  # Adjust multiplier as needed
                    
    #         # # Use a text area to display the content with adjustable height
    #         # st.text_area(
    #         #     label="",
    #         #     value=tagged_output.get('summary'),
    #         #     height=display_height,
    #         #     disabled=True
    #         # )

    # # Add a clear button at the bottom
    # if st.session_state.processed_results:
    #     if st.button("Clear All Results", type="secondary"):
    #         st.session_state.processed_results = []
    #         st.session_state.responses_df = pd.DataFrame(
    #             columns=['transcript', 'response', 'timestamp']
    #         )
    #         st.rerun()

    import streamlit as st
import os
import pandas as pd
from datetime import datetime
import json

def display_transcript_analysis():
    st.title("Sentinuum AI")

    # Initialize session state
    if 'processed_results' not in st.session_state:
        st.session_state.processed_results = []
    if 'responses_df' not in st.session_state:
        st.session_state.responses_df = pd.DataFrame(columns=['transcript', 'response', 'timestamp'])

    # === Dropdown to select transcript file ===
    TEXT_DIR = "C:/Users/niveditha.n.lv/Documents/summarizer/sentinuum v1/conversations"
    text_files = [f for f in os.listdir(TEXT_DIR) if f.endswith(".txt")]

    if not text_files:
        st.warning(f"No .txt files found in '{TEXT_DIR}' directory.")
        return

    selected_file = st.selectbox("Select a transcript file:", text_files)

    # Read content from selected file
    file_path = os.path.join(TEXT_DIR, selected_file)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            conversation_input = f.read()
    except Exception as e:
        st.error(f"Failed to read file: {e}")
        return

    # Show the transcript in a disabled text area
    st.text_area("Transcript Preview:", conversation_input, height=300, disabled=True)

    # === Process Button ===
    if st.button("Process Conversation", type="primary"):
        with st.status("Processing conversation...") as status:
            st.write("Analyzing conversation...")

            # Call your processing functions (define them elsewhere)
            tagged_output = analyze_conversation(conversation_input)
            result = process_transcript(tagged_output.get('tagged_output'))

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            result_with_metadata = {
                'transcript': conversation_input[:100] + "..." if len(conversation_input) > 100 else conversation_input,
                'processed_result': result,
                'tagged_output': tagged_output,
                'timestamp': timestamp
            }

            st.session_state.processed_results.append(result_with_metadata)
            status.update(label="✅ Processing complete!", state="complete")

        # === Display Results ===
        tab1, tab2 = st.tabs(["Transcript Breakdown", "Analysis Results"])
        with tab2:
            display_analysis_result(result)
        with tab1:
            display_beautified_transcript(tagged_output.get('summary'))

    # === Clear Button ===
    if st.session_state.processed_results:
        if st.button("Clear All Results", type="secondary"):
            st.session_state.processed_results = []
            st.session_state.responses_df = pd.DataFrame(columns=['transcript', 'response', 'timestamp'])
            st.rerun()



def display_beautified_transcript(tagged_output):
    """Display tagged transcript in a beautifully formatted way"""
    st.subheader("Conversation Analysis")
        
    try:
        # Parse the tagged output if it's a string
        if isinstance(tagged_output, str):
            try:
                data = json.loads(tagged_output)
            except json.JSONDecodeError:
                st.error("Unable to parse the tagged output as JSON")
                st.text_area("Raw Tagged Output", value=tagged_output, height=500)
                return
        else:
            data = tagged_output
            
        # Display Metadata in a nice card
        if 'conversation_metadata' or 'analytics' in data:
            metadata = data['conversation_metadata']
            # analytics = data['analytics']['overall_sentiment']
            analytics = data['analytics']
            sentiment = analytics.get('overall_sentiment', 'N/A')
            sentiment_color = get_sentiment_color(sentiment)
            st.markdown("### Metadata")
            # st.markdown(f"**Overall Sentiment:** {analytics.get('overall_sentiment', 'N/A')}")
            st.markdown(f"**Overall Sentiment:** <span style='color:{sentiment_color};'>{sentiment}</span>", 
                        unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Customer:** {metadata.get('customer_name', 'N/A')}")
                st.markdown(f"**Account Type:** {metadata.get('account_type', 'N/A')}")
                st.markdown(f"**Account Number:** {metadata.get('account_number', 'N/A')}")
            with col2:
                st.markdown(f"**Call Type:** {metadata.get('call_type', 'N/A')}")
                st.markdown(f"**Agent Name:** {metadata.get('agent_name', 'N/A')}")
            
        # Display Summary
        if 'conversation_details' in data and 'summary' in data['conversation_details']:
            summary = data['conversation_details']['summary']
            st.markdown("### High-level Summary")
            st.markdown(f"**Overview:** {summary.get('overview', 'N/A')}")
                
            # Summary details in columns
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Primary Issue:** {summary.get('primary_issue', 'N/A')}")
            with col2:
                st.markdown(f"**Resolution Status:** {summary.get('resolution_status', 'N/A')}")
                st.markdown(f"**Call Outcome:** {summary.get('call_outcome', 'N/A')}")
            
        # Display Transcript Segments as a timeline
        if 'conversation_details' in data and 'transcript_segments' in data['conversation_details']:
            segments = data['conversation_details']['transcript_segments']
            with st.expander("### Timeline", expanded=False):
                
                for idx, segment in enumerate(segments):
                    # Create a container for each message
                    with st.container():
                        # Determine if it's customer or agent for styling
                        is_customer = segment.get('speaker', '').lower() == 'customer'
                        speaker_color = "rgba(0, 120, 212, 0.1)" if is_customer else "rgba(100, 100, 100, 0.1)"
                        speaker_name = segment.get('speaker', 'Unknown').capitalize()
                            
                        # Format timestamp if available
                        timestamp_str = ""
                        if 'timestamp' in segment:
                            try:
                                # Convert ISO timestamp to readable format
                                dt = datetime.fromisoformat(segment['timestamp'].replace('Z', '+00:00'))
                                timestamp_str = dt.strftime("%H:%M:%S")
                            except:
                                timestamp_str = segment['timestamp']

                        st.markdown(
                            f"""
                                    <div style="
                                    background-color: {speaker_color}; 
                                    padding: 10px 15px; 
                                    border-radius: 10px; 
                                    margin: 5px 0;
                                    max-width: 80%;
                                    {'' if is_customer else 'margin-left: 20%;'}
                                    ">
                                    <strong>{speaker_name}</strong> <span style="color: gray; font-size: 0.8em;">({timestamp_str})</span>
                                    <p style="margin: 5px 0 0 0;">{segment.get('text', '')}</p>
                                    {f'<div style="font-size: 0.8em; margin-top: 5px;"><em>Sentiment: <span style="color: {"red" if segment.get("sentiment") == "negative" else "green" if segment.get("sentiment") == "positive" else "gray"}">{segment.get("sentiment", "neutral")}</span></em></div>' if 'sentiment' in segment else ''}
                                    {f'<div style="font-size: 0.8em; margin-top: 2px;"><em>Key phrases: {", ".join([f"<strong>{phrase}</strong>" for phrase in segment.get("key_phrases", [])])}</em></div>' if segment.get('key_phrases') else ''}
                                    </div>
                                    """, 
                                    unsafe_allow_html=True
                                    )
                
    except Exception as e:
            st.error(f"Error displaying beautified transcript: {str(e)}")
            # Fallback to raw display
            st.text_area("Raw Tagged Output", value=str(tagged_output), height=500)       
        
if __name__ == "__main__":     
    display_transcript_analysis()