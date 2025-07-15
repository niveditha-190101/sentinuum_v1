tagging_prompt = f"""
            Given the following conversation transcript between a customer and a bank agent, classify each line with the speaker.
            Rules for classification:
            1. Agents typically:
                - Introduce themselves and their company
                - Ask for verification information
                - Use formal language and company protocols
                - Provide solutions and explanations
                - End conversations with closing statements

              2. Customers typically:
                - Describe problems or make requests
                - Provide personal information when asked
                - Ask questions about services/issues
                - Respond to agent's questions
                - Express satisfaction/dissatisfaction

            Format each line as either 'Customer: <text>' or 'Agent: <text>'.

            Transcript:"""

extraction_prompt = f"""
        Analyze this conversation and generate a structured schema as per the following steps:

        1. Provide Metadata:
            - Participants (customer and agent)
            - Duration (in seconds)

        2. Generate a Summary:
            - Main issue (1-2 words)
            - Customer concerns
            - Resolution status: Understand the whole conversation and based on the end status, classify one among [resolved, unresolved, pending]

        3. Perform Sentiment Analysis:
            - Determine the overall sentiment trend of the entire conversation by analyzing the tone, language, and context
            - Classify the overall sentiment as:
                - 'positive': customer exhibits satisfaction, gratitude, or constructive engagement
                - 'neutral': conversation reflects a balanced tone with no strong emotions
                - 'negative': customer exhibits frustration, dissatisfaction, or complaints
            - Populate: \"analytics\": one among positive, neutral, negative"
            - Give only one sentiment based on whole conversation

        4. Extract Key Phrases (1-2 Words):
            - Identify concise key transaction details (e.g., \"contribution error\", \"fraudulent charge\")
            - Pinpoint issues or topics related to financial domain(e.g., \"account error\", \"policy confusion\")

        5. Identify Customer Concerns (1-2 Words):
            - Summarize customer concerns succinctly (e.g., \"fraud\", \"billing\", \"confusion\")

        6. Track Agent Actions (1-2 Words):
            - type should have one among \"follow_up_call\", \"document_verification\", \"internal_escalation\"
            - assigned_to should have either the name of the employee (the agent) or the relevant department
            - due_date should contain the date
            - status should have one among \"open\", \"in_progress\", \"completed\"

        7. Analytics:
            - The overall sentiment of the conversation

        Key Notes:
            - Ensure that all extracted content is limited to 1-2 words while still conveying the main meaning
            - The overall sentiment trend should reflect the tone of the entire conversation
            - Fill in all details. Do not leave any JSON fields of the schema blank

        Conversation: """