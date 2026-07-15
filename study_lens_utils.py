from __future__ import annotations

import os


def normalize_provider(provider: str | None) -> str:
    provider_name = (provider or "").strip().lower()
    aliases = {
        "google": "gemini",
        "gemini": "gemini",
        "gpt": "openai",
        "openai": "openai",
        "claude": "anthropic",
        "anthropic": "anthropic",
    }
    return aliases.get(provider_name, provider_name or "gemini")


def get_preferred_chat_models(provider: str | None = None) -> list[str]:
    provider_name = normalize_provider(provider)
    if provider_name == "openai":
        return ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"]
    if provider_name == "anthropic":
        return ["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"]
    return ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro-latest"]


def resolve_api_key(provider: str | None, supplied_key: str | None = None) -> str | None:
    provider_name = normalize_provider(provider)
    if supplied_key and supplied_key.strip():
        return supplied_key.strip()

    env_names = {
        "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "openai": ["OPENAI_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY"],
    }.get(provider_name, ["GEMINI_API_KEY", "GOOGLE_API_KEY"])

    for env_name in env_names:
        value = os.getenv(env_name)
        if value and value.strip():
            return value.strip()

    return None


def resolve_model_name(provider: str | None, supplied_model: str | None = None) -> str:
    provider_name = normalize_provider(provider)
    if supplied_model and supplied_model.strip():
        return supplied_model.strip()

    defaults = {
        "gemini": os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
        "openai": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "anthropic": os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
    }
    return defaults.get(provider_name, defaults["gemini"])


def create_chat_model(provider: str | None, api_key: str | None, model_name: str | None = None):
    provider_name = normalize_provider(provider)
    model = resolve_model_name(provider_name, model_name)

    if provider_name == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI support requires the langchain-openai package. Install it with pip install langchain-openai openai."
            ) from exc
        return ChatOpenAI(model=model, temperature=0, api_key=api_key)

    if provider_name == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise RuntimeError(
                "Anthropic support requires the langchain-anthropic package. Install it with pip install langchain-anthropic anthropic."
            ) from exc
        return ChatAnthropic(model=model, temperature=0, api_key=api_key)

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise RuntimeError(
            "Gemini support requires the langchain-google-genai package. Install it with pip install langchain-google-genai."
        ) from exc

    os.environ.setdefault("GOOGLE_API_KEY", api_key or "")
    return ChatGoogleGenerativeAI(model=model, temperature=0)


def create_embeddings(provider: str | None, api_key: str | None):
    provider_name = normalize_provider(provider)
    if provider_name == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI embeddings require the langchain-openai package. Install it with pip install langchain-openai openai."
            ) from exc
        return OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

    if provider_name == "anthropic":
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError as exc:
            raise RuntimeError(
                "Anthropic fallback embeddings require sentence-transformers and langchain-community."
            ) from exc
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
    except ImportError as exc:
        raise RuntimeError(
            "Gemini embeddings require the langchain-google-genai package. Install it with pip install langchain-google-genai."
        ) from exc

    os.environ.setdefault("GOOGLE_API_KEY", api_key or "")
    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


def is_quota_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        marker in message
        for marker in [
            "resource_exhausted",
            "quota",
            "rate limit",
            "429",
            "exceeded your current quota",
        ]
    )


def format_model_error(error: Exception, provider: str | None = None) -> str:
    provider_name = normalize_provider(provider)
    if is_quota_error(error):
        if provider_name == "openai":
            return (
                "The OpenAI API quota or rate limit has been exhausted. "
                "Please wait a moment or use a different API key."
            )
        if provider_name == "anthropic":
            return (
                "The Anthropic API quota or rate limit has been exhausted. "
                "Please wait a moment or use a different API key."
            )
        return (
            "The Gemini API quota has been exhausted for your account. "
            "Please wait a few minutes or switch to a paid plan before asking another question."
        )

    return f"Could not answer the question: {error}"
