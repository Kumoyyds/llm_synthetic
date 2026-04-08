import pandas as pd
import json
import os

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI  # Requires langchain-openai package
from dotenv import load_dotenv
load_dotenv()
lite_llm_key_all = os.getenv('LITE_LLM_KEY_ALL')

llm_model = ChatOpenAI(base_url='https://ipsos.litellm-prod.ai/',model='gpt-5', temperature=0.1, api_key=lite_llm_key_all)

from typing import Literal, Union, Dict

KPI_TYPE = Literal["relevance", "differentiation", "believability"]

KPI_QUESTIONS = {
    "relevance": {
        "question": "Is this product concept relevant to your needs and interests?",
        "criteria": """
        - Does this concept address needs, problems, or desires mentioned in the consumer insights?
        - Does it fit their lifestyle, preferences, or values?
        - Would they find this product useful or appealing based on their stated needs?
        """
    },
    "differentiation": {
        "question": "Does this product concept stand out as unique or different from what's currently available?",
        "criteria": """
        - Does this concept offer something new or innovative that would catch attention?
        - Is it meaningfully different from existing products the consumer might know?
        - Would this concept make the consumer want to try it over alternatives?
        """
    },
    "believability": {
        "question": "Do you find this product concept believable and credible?",
        "criteria": """
        - Are the claims and benefits stated in the concept realistic and trustworthy?
        - Does the concept align with what the consumer knows or expects from this category?
        - Would the consumer believe the product can deliver on its promises?
        """
    }
}

def ai_filter(
    new_concept: str, 
    kpi_type: KPI_TYPE, 
    system_info: dict, 
    llm = llm_model,
    return_reasoning: bool = False
) -> Union[str, dict]:
    """
    Evaluate a product concept from a consumer's perspective.
    
    Args:
        new_concept: The product concept text to evaluate
        kpi_type: One of "relevance", "differentiation", or "believability"
        system_info: Dict containing consumer insights (qneed2, qneed3, insight)
        llm: LLM model instance
        return_reasoning: If True, return dict with 'answer' and 'reasoning'
        
    Returns:
        If return_reasoning=False: "yes" or "no"
        If return_reasoning=True: {"answer": "yes/no", "reasoning": "..."}
    """
    
    # Get KPI-specific question and criteria
    kpi_config = KPI_QUESTIONS[kpi_type]
    
    # Prepare consumer insights
    qneeds = []
    for q in ['qneed2', 'qneed3']:
        if q in system_info:
            qneeds.append(system_info[q])
    qneeds = [f"{q['cate']}: {q['comment']}" for q in qneeds]
    qneeds_text = "\n".join(qneeds) if qneeds else "No specific category needs provided."
    insight_text = system_info.get('insight', 'No general insights provided.')
    
    system_prompt = f"""You are a digital twin mimicking a specific consumer. Your role is to respond as this consumer would, based on their interview insights.

Consumer Profile:
=================
Category-Specific Needs:
{qneeds_text}

General Insights & Preferences:
{insight_text}
=================

Respond authentically as this consumer would, based on their actual needs, values, and preferences."""

    if return_reasoning:
        user_query = f"""Evaluate this product concept:
---
{new_concept}
---

Question: {kpi_config['question']}

Evaluation Criteria:
{kpi_config['criteria']}

Think step by step:
1. Review the consumer's specific needs and general insights
2. Assess the concept against each evaluation criterion
3. Determine if this consumer would answer yes or no

Output format (use exactly this format):
REASONING: <your step-by-step reasoning here>
ANSWER: <yes or no>"""
    else:
        user_query = f"""Evaluate this product concept:
---
{new_concept}
---

Question: {kpi_config['question']}

Evaluation Criteria:
{kpi_config['criteria']}

Think step by step:
1. Review the consumer's specific needs and general insights
2. Assess the concept against each evaluation criterion
3. Determine if this consumer would answer yes or no

Output ONLY "yes" or "no" (lowercase, no punctuation or explanation)."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_query)
    ]
    
    ai_response = llm.invoke(messages)
    response_text = ai_response.content.strip()
    
    if return_reasoning:
        # Parse reasoning and answer
        reasoning = ""
        answer = ""
        if "REASONING:" in response_text and "ANSWER:" in response_text:
            parts = response_text.split("ANSWER:")
            reasoning = parts[0].replace("REASONING:", "").strip()
            answer = parts[1].strip().lower()
        else:
            # Fallback: try to extract yes/no from the end
            answer = "yes" if "yes" in response_text.lower() else "no"
            reasoning = response_text
        return {"answer": answer, "reasoning": reasoning}
    else:
        return response_text.lower()
