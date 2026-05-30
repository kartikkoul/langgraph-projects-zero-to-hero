from abc import ABC

from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.schemas import IntentSchema, QualityScoreSchema, SentimentSchema, UrgencySchema

str_parser = StrOutputParser()

"""
All the prompt templates are put here.

Every Prompt Class comes with two attributes:

1. parser
2. prompt_template
"""


class PromptBaseClass(ABC):
    parser: PydanticOutputParser | StrOutputParser
    prompt_template: ChatPromptTemplate


class IntentAnalyzerPrompt(PromptBaseClass):
    def __init__(self) -> None:
        super().__init__()

    parser = PydanticOutputParser(pydantic_object=IntentSchema)

    prompt_template = ChatPromptTemplate(
        messages=[
            (
                "system",
                """
                You are an intent analyzer bot that classifies the user's support query.

                Examples:
                Q: "My payment failed but money got deducted."
                Intent: billing_agent
                Q: "The app crashes when I upload a photo."
                Intent: technical_agent
                Q: "I want to cancel my subscription."
                Intent: billing_agent
                Q: "How do I reset my password?"
                Intent: general_agent
                Q: "Your API is returning 500 errors."
                Intent: technical_agent
                Q: "Where can I download my invoice?"
                Intent: billing_agent
                Q: "How much does the premium plan cost?"
                Intent: billing_agent
                Q: "The website feels very slow."
                Intent: technical_agent
                Q: "Can you explain your features?"
                Intent: general_agent
                Q: "Thank you for the support."
                Intent: general_agent

                {format_instructions}
                """,
            ),
            ("human", "Analyze the intent of the query: {user_query}"),
        ],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )


class SentimentAnalyzerPrompt(PromptBaseClass):
    def __init__(self) -> None:
        super().__init__()

    parser = PydanticOutputParser(pydantic_object=SentimentSchema)

    prompt_template = ChatPromptTemplate(
        messages=[
            (
                "system",
                """
                You are a sentiment analyzer bot that classifies the user's emotional tone.

                Examples:
                Q: "I've emailed support 5 times and nobody replied."
                Sentiment: frustrated
                Q: "This is the worst experience I've had."
                Sentiment: angry
                Q: "Thanks, everything works perfectly now."
                Sentiment: positive
                Q: "Can someone help me with login?"
                Sentiment: neutral
                Q: "Amazing support team. Really appreciate it."
                Sentiment: happy
                Q: "I'm disappointed with the recent update."
                Sentiment: negative
                Q: "I'm confused about the pricing."
                Sentiment: confused
                Q: "Please fix this ASAP. I'm extremely upset."
                Sentiment: furious
                Q: "No worries, take your time."
                Sentiment: calm
                Q: "I'm worried my account was hacked."
                Sentiment: anxious

                {format_instructions}
                """,
            ),
            ("human", "Analyze the sentiment of the query: {user_query}"),
        ],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )


class UrgencyAnalyzerPrompt(PromptBaseClass):
    def __init__(self) -> None:
        super().__init__()

    parser = PydanticOutputParser(pydantic_object=UrgencySchema)

    prompt_template = ChatPromptTemplate(
        messages=[
            (
                "system",
                """
                    You are an urgency analyzer bot that scores how urgent the user's issue is from 1 (low) to 10 (critical).

                    Examples:
                    Q: "Our production server is down!"
                    Urgency: 10
                    Q: "My account may have been hacked."
                    Urgency: 9
                    Q: "Payment failed during checkout."
                    Urgency: 8
                    Q: "Nobody in my company can log in."
                    Urgency: 10
                    Q: "I need this fixed before today's meeting."
                    Urgency: 8
                    Q: "The app crashes sometimes."
                    Urgency: 6
                    Q: "Just wanted to report a UI bug."
                    Urgency: 4
                    Q: "The website feels a bit slow."
                    Urgency: 3
                    Q: "I found a typo on the homepage."
                    Urgency: 1
                    Q: "Whenever convenient, can you check this?"
                    Urgency: 2

                    {format_instructions}
                    """,
            ),
            ("human", "Analyze the urgency of the query: {user_query}"),
        ],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )


class AgentAssignmentPrompt(PromptBaseClass):
    def __init__(self) -> None:
        super().__init__()

    parser = str_parser

    prompt_template = ChatPromptTemplate(
        messages=[
            (
                "system",
                """
                    Deliver a short message telling the user they are being directed to {intent},
                    acknowledging their {sentiment} tone and urgency level {urgency}.

                    Examples:
                    Q: "My payment failed but money got deducted." | Intent: billing_agent | Sentiment: frustrated | Urgency: 8
                    A: We understand how frustrating this is. Your issue is high priority and we are connecting you with a billing agent right away.

                    Q: "The app crashes whenever I upload an image." | Intent: technical_agent | Sentiment: angry | Urgency: 7
                    A: We apologize for the inconvenience. A technical agent is being assigned to investigate promptly.

                    Q: "I want to know how to upgrade my plan." | Intent: general_agent | Sentiment: neutral | Urgency: 2
                    A: Thank you for reaching out. A general support agent will help you with upgrade options.

                    Q: "Nobody in my company can log in right now." | Intent: technical_agent | Sentiment: anxious | Urgency: 10
                    A: We understand the urgency. Your request is escalated and a technical agent is assigned with top priority.
                    """,
            ),
            MessagesPlaceholder("messages"),
            ("human", "Write the routing message for this customer."),
        ],
    )



class TechnicalAgentPrompt(PromptBaseClass):
    def __init__(self) -> None:
        super().__init__()

    parser = str_parser

    prompt_template = ChatPromptTemplate(
        messages=[
            (
                "system",
                """
                    You are a technical agent who responds to technical queries only.
                    """,
            ),
            MessagesPlaceholder("messages"),
            ("human", "Address to the user's query & provide a solution"),
        ],
    )


class BillingAgentPrompt(PromptBaseClass):
    def __init__(self) -> None:
        super().__init__()

    parser = str_parser

    prompt_template = ChatPromptTemplate(
        messages=[
            (
                "system",
                """
                    You are a billing agent who responds to billing queries only.
                    """,
            ),
            MessagesPlaceholder("messages"),
            ("human", "Address to the user's query & provide a solution"),
        ],
    )


class GeneralAgentPrompt(PromptBaseClass):
    def __init__(self) -> None:
        super().__init__()

    parser = str_parser

    prompt_template = ChatPromptTemplate(
        messages=[
            (
                "system",
                """
                    You are a generalist agent who responds to general queries.
                    """,
            ),
            MessagesPlaceholder("messages"),
            ("human", "Address to the user's query."),
        ],
    )

class ResponseQualityCheckerPrompt(PromptBaseClass):
    def __init__(self) -> None:
        super().__init__()

    parser = PydanticOutputParser(pydantic_object=QualityScoreSchema)

    prompt_template = ChatPromptTemplate(
        messages = [
            ("system", """You are a quality checker of the final response that scores based on the quality of the response \n {format_instructions} Is the response good?
                        Based on parameters like:
                        Is it polite?
                        Is it complete?
                        Is urgency acknowledged?
                        """),
            MessagesPlaceholder("messages"),
            ("human", "Score the final response: {final_response}")
        ],

        partial_variables={
            "format_instructions": parser.get_format_instructions()
        }
    )