from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from adgen.models.ad import AdConcept, AdProject, AdScript, VisualPlan
from adgen.utils.config import Config, get_api_key


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
    workflow.add_node("generate_concept", generate_concept_node)
    workflow.add_node("review_concept", review_concept_node)
    workflow.add_node("generate_script", generate_script_node)
    workflow.add_node("generate_visual_plan", generate_visual_plan_node)

    # Add edges
    workflow.set_entry_point("generate_concept")
    workflow.add_edge("generate_concept", "review_concept")
    workflow.add_conditional_edges(
        "review_concept",
        should_continue_to_media,
        {"generate_media": "generate_script", END: END},
    )
    workflow.add_edge("generate_script", "generate_visual_plan")
    workflow.add_edge("generate_visual_plan", END)

    return workflow.compile()
