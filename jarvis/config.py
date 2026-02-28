import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent

class Config:
    # Ollama
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    # Smaller/faster model for internal reasoning (metacognition, critic, planner).
    # Falls back to OLLAMA_MODEL if not set or not available.
    OLLAMA_FAST_MODEL: str = os.getenv("OLLAMA_FAST_MODEL", "")

    # Voice
    VOICE_ENABLED: bool = os.getenv("VOICE_ENABLED", "false").lower() == "true"
    VOICE_STT_ENGINE: str = os.getenv("VOICE_STT_ENGINE", "whisper")
    VOICE_WHISPER_MODEL: str = os.getenv("VOICE_WHISPER_MODEL", "base")
    VOICE_TTS_ENGINE: str = os.getenv("VOICE_TTS_ENGINE", "pyttsx3")

    # Memory
    MEMORY_MAX_SHORT_TERM: int = int(os.getenv("MEMORY_MAX_SHORT_TERM", "20"))
    MEMORY_CHROMA_PATH: str = os.getenv("MEMORY_CHROMA_PATH", str(BASE_DIR / "data" / "memory"))
    # Consolidate long-term memory every N interactions (0 = disabled)
    MEMORY_CONSOLIDATION_INTERVAL: int = int(os.getenv("MEMORY_CONSOLIDATION_INTERVAL", "20"))

    # Code execution
    CODE_EXEC_TIMEOUT: int = int(os.getenv("CODE_EXEC_TIMEOUT", "10"))
    CODE_EXEC_ENABLED: bool = os.getenv("CODE_EXEC_ENABLED", "true").lower() == "true"
    CODE_DEBUG_RETRIES: int = int(os.getenv("CODE_DEBUG_RETRIES", "3"))

    # Critic
    CRITIC_ENABLED: bool = os.getenv("CRITIC_ENABLED", "true").lower() == "true"

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_PATH: str = os.getenv("LOG_PATH", str(BASE_DIR / "data" / "logs" / "jarvis.log"))

    # Self-improvement
    REFLECTION_INTERVAL: int = int(os.getenv("REFLECTION_INTERVAL", "5"))
    SELF_IMPROVE_ENABLED: bool = os.getenv("SELF_IMPROVE_ENABLED", "true").lower() == "true"

    # Context window compression
    # When short-term memory is near-full, the oldest half of turns is summarised
    # into a compact system message to prevent silent context loss.
    CONTEXT_COMPRESS_ENABLED: bool = os.getenv("CONTEXT_COMPRESS_ENABLED", "true").lower() == "true"
    # Compression triggers when the buffer is within this many messages of its limit.
    CONTEXT_COMPRESS_THRESHOLD_OFFSET: int = int(os.getenv("CONTEXT_COMPRESS_THRESHOLD_OFFSET", "4"))

    # Proactive web search
    # When pre-flight confidence is below this threshold on a factual query,
    # Jarvis automatically searches before generating a response.
    PROACTIVE_SEARCH_ENABLED: bool = os.getenv("PROACTIVE_SEARCH_ENABLED", "true").lower() == "true"
    PROACTIVE_SEARCH_CONFIDENCE_THRESHOLD: float = float(
        os.getenv("PROACTIVE_SEARCH_CONFIDENCE_THRESHOLD", "0.35")
    )

    # Paths
    SKILLS_PATH: Path = BASE_DIR / "data" / "skills"
    SELF_MODEL_PATH: Path = BASE_DIR / "data" / "self_model.json"
    META_PROMPT_PATH: Path = BASE_DIR / "data" / "meta_prompt.txt"
    INTROSPECTION_PATH: Path = BASE_DIR / "data" / "introspection_state.json"

config = Config()
