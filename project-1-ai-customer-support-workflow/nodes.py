from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from prompts import (
    AgentAssignmentPrompt,
    BillingAgentPrompt,
    GeneralAgentPrompt,
    IntentAnalyzerPrompt,
    ResponseQualityCheckerPrompt,
    SentimentAnalyzerPrompt,
    TechnicalAgentPrompt,
    UrgencyAnalyzerPrompt,
)
from state_schemas import CustomerAgentState

load_dotenv()


def initiate_model(state: CustomerAgentState):
    user_query = state["user_query"]
    model = ChatOpenAI(model="gpt-4.1-2025-04-14", temperature=0.5)

    return {
        "model": model,
        "user_query": user_query,
        "messages": [HumanMessage(content=user_query)],
        "retry_count": 1
    }


def analyze_intent(state: CustomerAgentState):
    model = state["model"]

    prompt = IntentAnalyzerPrompt()
    user_query = state["user_query"]

    intent_analyzer = prompt.prompt_template | model.bind(max_tokens=50) | prompt.parser
    result = intent_analyzer.invoke({"user_query": user_query})

    return {"intent": result.intent}


def analyze_sentiment(state: CustomerAgentState):
    model = state["model"]

    prompt = SentimentAnalyzerPrompt()
    user_query = state["user_query"]

    sentiment_analyzer = prompt.prompt_template | model.bind(max_tokens=50) | prompt.parser
    result = sentiment_analyzer.invoke({"user_query": user_query})

    return {"sentiment": result.sentiment}


def analyze_urgency(state: CustomerAgentState):
    model = state["model"]

    prompt = UrgencyAnalyzerPrompt()
    user_query = state["user_query"]

    urgency_analyzer = prompt.prompt_template | model.bind(max_tokens=50) | prompt.parser
    result = urgency_analyzer.invoke({"user_query": user_query})

    return {"urgency": result.urgency}


def assign_respective_agent(state: CustomerAgentState):
    model = state["model"]

    prompt = AgentAssignmentPrompt()

    agent_assigner = prompt.prompt_template | model | prompt.parser

    response = agent_assigner.invoke(
        {
            "intent": state["intent"],
            "sentiment": state["sentiment"],
            "urgency": state["urgency"],
            "messages": state["messages"],
        }
    )

    return {"messages": [AIMessage(content=response)]}


def technical_agent(state: CustomerAgentState):
    model = state["model"]
    prompt = TechnicalAgentPrompt()

    technical_agent = prompt.prompt_template | model | prompt.parser

    draft_response = technical_agent.invoke({
        "messages": state["messages"]
    })


    return {"draft_response": draft_response}


def billing_agent(state: CustomerAgentState):
    model = state["model"]
    prompt = BillingAgentPrompt()

    billing_agent = prompt.prompt_template | model | prompt.parser

    draft_response = billing_agent.invoke({"messages": state["messages"]})
    


    return {"draft_response": draft_response}


def general_agent(state: CustomerAgentState):
    model = state["model"]
    prompt = GeneralAgentPrompt()

    general_agent = prompt.prompt_template | model | prompt.parser

    draft_response = general_agent.invoke({"messages": state["messages"]})
    

    return {"draft_response": draft_response}

def check_response_quality(state: CustomerAgentState):
    model = state["model"]
    prompt = ResponseQualityCheckerPrompt()

    quality_scorer = prompt.prompt_template | model | prompt.parser

    result = quality_scorer.invoke({
        "messages": state["messages"],
        "final_response": state["draft_response"]
    })
    

    return {"final_response_quality_score": result.quality_score , "retry_count": state["retry_count"]+1}

def final_response(state: CustomerAgentState):
    return {"final_response": state["draft_response"]}