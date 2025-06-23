from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from adgen.models.ad import AdConcept, AdProject, AdScript, VisualPlan
from adgen.utils.config import Config, get_api_key
from adgen.utils.markdown import get_markdown_from_web_page


class AdGenerationState(dict[str, Any]):
    """State object for the ad generation workflow."""

    project: AdProject
    config: Config
    approve_concept: bool = False
    approve_final: bool = False


def create_llm(config: Config) -> BaseChatModel:
    """Create LLM instance based on configuration."""
    provider = config.providers["llm"]
    api_key = get_api_key(provider, "llm")

    if not api_key:
        raise ValueError(f"API key not found for {provider}")

    if provider == "openai":
        model = config.llm.get("openai", {}).get("model", "gpt-4")
        temperature = config.llm.get("openai", {}).get("temperature", 0.7)
        return ChatOpenAI(api_key=api_key, model=model, temperature=temperature)
    elif provider == "anthropic":
        model = config.llm.get("anthropic", {}).get("model", "claude-3-sonnet-20240229")
        temperature = config.llm.get("anthropic", {}).get("temperature", 0.7)
        return ChatAnthropic(api_key=api_key, model=model, temperature=temperature)
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


async def scrape_web_content_node(state: AdGenerationState) -> AdGenerationState:
    """Extract markdown content from the provided URL."""
    if not state["project"].source_url:
        # Skip if no URL provided
        return state

    print(f"Scraping content from: {state['project'].source_url}")

    try:
        markdown_content = await get_markdown_from_web_page(state["project"].source_url)
        state["project"].web_content_markdown = markdown_content
        state["project"].status = "web_scraped"
        print(f"Successfully scraped {len(markdown_content)} characters of content")
    except Exception as e:
        print(f"Error scraping web content: {e}")
        # Continue with empty content for now
        state["project"].web_content_markdown = ""
        state["project"].status = "web_scrape_failed"

    return state


async def generate_business_description_node(
    state: AdGenerationState,
) -> AdGenerationState:
    """Generate a business description from scraped web content."""
    if not state["project"].web_content_markdown:
        # Skip if no web content
        return state

    llm = create_llm(state["config"])

    prompt = f"""
    You are a business analyst. Analyze the following website content and create a comprehensive business description that will be used for creating a video advertisement.

    Website Content (Markdown):
    {state["project"].web_content_markdown[:8000]}  # Limit to avoid token limits

    Please create a detailed business description that includes:

    ## Company Overview
    - What kind of business this is
    - Their mission and values
    - Their unique value proposition

    ## Branding & Style
    - Brand personality and tone
    - Visual style preferences (if evident)
    - Brand positioning

    ## Products & Services
    - Main product categories
    - Key products or services
    - Target market for each

    ## Customer Focus
    - Primary customer demographics
    - Customer pain points they solve
    - Customer success stories or testimonials (if mentioned)

    Format this as a well-structured markdown document that captures the essence of the business for advertising purposes. Focus on details that would be relevant for creating a compelling video advertisement.
    """

    print("Generating business description from web content...")

    try:
        description = await llm.ainvoke(prompt)
        state["project"].business_description = description.content
        state["project"].status = "business_analyzed"
        print(f"Generated business description: {len(description.content)} characters")
    except Exception as e:
        print(f"Error generating business description: {e}")
        # Fallback to truncated web content
        state["project"].business_description = state["project"].web_content_markdown[
            :2000
        ]
        state["project"].status = "business_analysis_failed"

    return state


def should_scrape_web_content(state: AdGenerationState) -> str:
    """Decide whether to scrape web content or go directly to concept generation."""
    if state["project"].source_url:
        return "scrape_web"
    elif state["project"].business_description:
        return "generate_concept"
    else:
        raise ValueError("Either source_url or business_description must be provided")


def should_generate_business_description(state: AdGenerationState) -> str:
    """Decide whether to generate business description from web content."""
    if (
        state["project"].web_content_markdown
        and state["project"].status == "web_scraped"
    ):
        return "generate_business_description"
    else:
        return "generate_concept"


async def generate_concept_node(state: AdGenerationState) -> AdGenerationState:
    """Generate the ad concept from business description."""
    llm = create_llm(state["config"]).with_structured_output(AdConcept)

    prompt = f"""
    You are an expert advertising strategist. Create a comprehensive ad concept for the following business:

    Business Description: {state["project"].business_description}

    Target Duration: {state["config"].ad_duration_seconds} seconds

    Please provide a detailed ad concept including:
    - Target audience
    - Key message
    - Tone
    - Visual style
    - Call to action
    - Emotional appeal

    Remember that we are trying to sell the products or services of the business. The target
    audience for the ad depends on the who the business is and what they sell, but is whoever
    the business sells to.

    Make it compelling and appropriate for a {state["config"].ad_duration_seconds}-second video ad.
    """

    print(f"Generating concept with prompt: {prompt}")

    concept = await llm.ainvoke(prompt)

    state["project"].concept = concept
    state["project"].status = "concept_generated"

    print(f"Concept generated: {concept}")
    return state


async def review_concept_node(state: AdGenerationState) -> AdGenerationState:
    """Human review of the generated concept."""
    # This will be handled by the CLI interface
    # For now, assume approval
    state["approve_concept"] = True
    return state


async def generate_script_node(state: AdGenerationState) -> AdGenerationState:
    """Generate the ad script based on the approved concept."""
    if not state["approve_concept"]:
        return state

    llm = create_llm(state["config"]).with_structured_output(AdScript)
    concept = state["project"].concept

    prompt = f"""
    Create a compelling {state["config"].ad_duration_seconds}-second video ad script based on this concept:

    Target Audience: {concept.target_audience}
    Key Message: {concept.key_message}
    Tone: {concept.tone}
    Call to Action: {concept.call_to_action}

    The script should:
    - Have a strong hook in the first 3 seconds
    - Deliver the key message clearly
    - End with a clear call to action
    - Be exactly the right length for {state["config"].ad_duration_seconds} seconds (approximately {state["config"].ad_duration_seconds * 2.5} words)

    Format the response with:
    Hook: [opening line]
    Main Content: [body of script]
    Call to Action: [closing]
    """

    print(f"Generating script with prompt: {prompt}")

    script = await llm.ainvoke(prompt)

    state["project"].script = script
    state["project"].status = "script_generated"

    print(f"Script generated: {script}")
    return state


async def generate_visual_plan_node(state: AdGenerationState) -> AdGenerationState:
    """Generate visual plan for the advertisement."""
    if not state["approve_concept"]:
        return state

    llm = create_llm(state["config"]).with_structured_output(VisualPlan)
    concept = state["project"].concept
    script = state["project"].script

    prompt = f"""
    Create a detailed visual plan for a {state["config"].ad_duration_seconds}-second video ad:

    Concept Style: {concept.style}
    Script: {script.hook} {script.main_content} {script.call_to_action}

    Provide:
    - 3-5 specific visual scenes/shots
    - Overall visual style description
    - Color palette (3-5 colors)
    - Text overlays to display
    """

    print(f"Generating visual plan with prompt: {prompt}")

    visual_plan = await llm.ainvoke(prompt)

    state["project"].visual_plan = visual_plan
    state["project"].status = "visual_plan_generated"

    print(f"Visual plan generated: {visual_plan}")
    return state


def should_continue_to_media(state: AdGenerationState) -> str:
    """Decide whether to continue to media generation or end."""
    if state["approve_concept"]:
        return "generate_media"
    else:
        return END


def create_ad_generation_workflow(_config: Config) -> StateGraph:
    """Create the ad generation workflow using LangGraph."""

    workflow = StateGraph(AdGenerationState)

    # Add nodes
    workflow.add_node("scrape_web_content", scrape_web_content_node)
    workflow.add_node(
        "generate_business_description", generate_business_description_node
    )
    workflow.add_node("generate_concept", generate_concept_node)
    workflow.add_node("review_concept", review_concept_node)
    workflow.add_node("generate_script", generate_script_node)
    workflow.add_node("generate_visual_plan", generate_visual_plan_node)

    # Set entry point and routing
    workflow.set_entry_point("route_input")
    workflow.add_node("route_input", lambda state: state)  # Dummy node for routing

    # Route based on input type
    workflow.add_conditional_edges(
        "route_input",
        should_scrape_web_content,
        {"scrape_web": "scrape_web_content", "generate_concept": "generate_concept"},
    )

    # Web scraping path
    workflow.add_conditional_edges(
        "scrape_web_content",
        should_generate_business_description,
        {
            "generate_business_description": "generate_business_description",
            "generate_concept": "generate_concept",
        },
    )

    # Business description generation to concept
    workflow.add_edge("generate_business_description", "generate_concept")

    # Concept review and continuation
    workflow.add_edge("generate_concept", "review_concept")
    workflow.add_conditional_edges(
        "review_concept",
        should_continue_to_media,
        {"generate_media": "generate_script", END: END},
    )
    workflow.add_edge("generate_script", "generate_visual_plan")
    workflow.add_edge("generate_visual_plan", END)

    return workflow.compile()
