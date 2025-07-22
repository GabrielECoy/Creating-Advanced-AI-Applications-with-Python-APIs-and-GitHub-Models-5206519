from typing import Tuple, Dict
import dotenv
import os
from dotenv import load_dotenv
import requests
import json
import streamlit as st
import os
from openai import OpenAI
from langsmith import wrappers, traceable

token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.inference.ai.azure.com"
model_name = "gpt-4o-mini"

client = OpenAI(
    base_url=endpoint,
    api_key=token,
)

load_dotenv()
EXCHANGERATE_API_KEY = os.getenv('EXCHANGERATE_API_KEY')

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "moneychanger"

@traceable
def get_exchange_rate(base: str, target: str, amount: str) -> Tuple:
    """Return a tuple of (base, target, amount, conversion_result (2 decimal places))"""
    url = f"https://v6.exchangerate-api.com/v6/{EXCHANGERATE_API_KEY}/pair/{base}/{target}/{amount}"
    response = json.loads(requests.get(url).text)
    return (base, target, amount, f'{response["conversion_result"]:.2f}')

@traceable
def call_llm(textbox_input, function_result=None) -> Dict:
    """Make a call to the LLM with the textbox_input as the prompt.
       If function_result is provided, include it in the conversation to get a natural language response."""
    
    tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "exchange_rate_function",
                        "description": "Convert a given amount of money from one currency to another. Each currency will be represented as a 3-letter code",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "base": {
                                    "type": "string",
                                    "description": "The base or original currency.",
                                },
                                "target": {
                                    "type": "string",
                                    "description": "The target or converted currency",
                                },
                                "amount": {
                                    "type": "string",
                                    "description": "The amount of money to convert from the base currency.",
                                },
                            },
                            "required": ["base", "target", "amount"],
                            "additionalProperties": False,
                        },
                    },
                }
                ]

    # Build messages list
    messages = [
        {
            "role": "system",
            "content": "You are a helpful currency exchange assistant. When you receive exchange rate data, provide a natural, conversational response about the conversion.",
        },
        {
            "role": "user",
            "content": textbox_input,
        }
    ]
    
    # If we have function result, add it to the conversation
    if function_result:
        messages.append({
            "role": "assistant",
            "content": f"I've calculated the exchange rate for you: {function_result['base']} {function_result['amount']} converts to {function_result['target']} {function_result['conversion_result']}. Let me provide you with a detailed response."
        })

    try:
        # If we have function result, don't use tools (we're getting the final response)
        if function_result:
            response = client.chat.completions.create(
                messages=messages,
                temperature=1.0,
                top_p=1.0,
                max_tokens=1000,
                model=model_name,
            )
        else:
            # First call with tools to extract parameters
            response = client.chat.completions.create(
                messages=messages,
                temperature=1.0,
                top_p=1.0,
                max_tokens=1000,
                model=model_name,
                tools=tools,
            )

    except Exception as e:
        print(f"Exception {e} for {textbox_input}")
        return None
    else:
        return response

@traceable
def run_pipeline(user_input):
    """Based on textbox_input, determine if you need to use the tools (function calling) for the LLM.
    Call get_exchange_rate(...) if necessary, then send result back to LLM for natural language response"""

    # First LLM call to extract parameters
    response = call_llm(user_input)
    
    if not response:
        st.write("Sorry, there was an error processing your request.")
        return
    
    if response.choices[0].finish_reason == "tool_calls":
        # Extract parameters from function call
        response_arguments = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        base = response_arguments["base"]
        target = response_arguments["target"]
        amount = response_arguments["amount"]
        
        # Get exchange rate
        _, _, _, conversion_result = get_exchange_rate(base, target, amount)
        
        # Prepare function result for LLM
        function_result = {
            "base": base,
            "target": target,
            "amount": amount,
            "conversion_result": conversion_result
        }
        
        # Send result back to LLM for natural language response
        final_response = call_llm(user_input, function_result)
        
        if final_response:
            st.write(final_response.choices[0].message.content)
        else:
            # Fallback to simple output if LLM call fails
            st.write(f'{base} {amount} is {target} {conversion_result}')
            
    elif response.choices[0].finish_reason == "stop":
        # No function calling needed, just display the response
        st.write(response.choices[0].message.content)
    else:
        st.write("Sorry, I couldn't process that request.")

# Title of the app
st.title("Multilingual Money Changer with natural responses")

# Text box for user input
user_input = st.text_input("Enter the amount and the currency")

# Submit button
if st.button("Submit"):
    # Display the input text below the text box
    run_pipeline(user_input)