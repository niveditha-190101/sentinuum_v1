# edit this
schema = {
  "type": "object",
  "properties": {
    "conversation_metadata": {
      "type": "object",
      "properties": {
    #     "customer_id": {
    #       "type": "number"
    #     },
        "timestamp": {
          "type": "string",
          "format": "date-time"
        },
        "call_type": {
          "type": "string",
          "enum": ["inbound", "outbound"]
        },
        "customer_name": {
          "type": "string"
        },
        "account_type": {
          "type": "string",
          "enum": [
            "Checking accounts",
            "Savings accounts",
            "Money market accounts (MMAs)",
            "Certificate of deposit (CD) accounts"
          ]
        },
        "agent_name": {
          "type": "string"
        },
        "account_number": {
          "type": "string"
        }
      },
      "required": ["timestamp", "call_type", "customer_name", "account_type"]
    },
    "conversation_details": {
      "type": "object",
      "properties": {
        "summary": {
          "type": "object",
          "properties": {
            "overview": {
              "type": "string"
            },
            "primary_issue": {
              "type": "string"
            },
            "resolution_status": {
              "type": "string",
              "enum": ["resolved", "pending", "escalated"]
            },
            "call_outcome": {
              "type": "string",
              "enum": ["resolved", "escalated", "callback_scheduled", "unresolved"]
            }
          },
          "required": ["overview", "primary_issue", "resolution_status", "call_outcome"]
        },
        "transcript_segments": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
            #   "utterance_id": {
            #     "type": "string"
            #   },
              "timestamp": {
                "type": "string",
                "format": "date-time"
              },
              "speaker": {
                "type": "string",
                "enum": ["customer", "agent"]
              },
              "text": {
                "type": "string"
              },
              "sentiment": {
                "type": "string",
                "enum": ["positive", "neutral", "negative"]
              },
              "key_phrases": {
                "type": "array",
                "items": {
                  "type": "string"
                }
              }
            },
            "required": ["timestamp", "sentiment", "speaker", "text", "key_phrases"]
          }
        }
      },
      "required": ["summary", "transcript_segments"]
    },
    "key_points": {
      "type": "object",
      "properties": {
        "customer_concerns": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "type": {
                "type": "string",
                "enum": ["account_issue","fraud", "transaction_dispute", "loan_inquiry", "credit_card_issue", "general_inquiry"]
              },
              "description": {
                "type": "string"
              },
              "transaction_reference": {
                "type": "object",
                "properties": {
                  "amount": {
                    "type": "number"
                  },
                  "date": {
                    "type": "string",
                    "format": "date"
                  },
                  "account_last4": {
                    "type": "string"
                  }
                }
              }
            },
            "required": ["type", "description"]
          }
        }
      }
    },
    "action_items": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "type": {
            "type": "string",
            "enum": ["follow_up_call", "document_verification", "internal_escalation"]
          },
          "action_type": {
            "type": "string",
            "enum": ["follow_up", "issue_resolution", "document_submission", "callback", "refund_process"]
          },
          "assigned_to": {
            "type": "string"
          },
          "due_date": {
            "type": "string",
            "format": "date-time"
          },
          "status": {
            "type": "string",
            "enum": ["open", "in_progress", "completed"]
          }
        },
        "required": ["type", "action_type", "assigned_to", "due_date", "status"]
      }
    },
    "analytics": {
      "type": "object",
      "properties": {
        "overall_sentiment": {
          "type": "string",
          "enum": ["positive", "neutral", "negative"]
        }
      }
    }
  }
}