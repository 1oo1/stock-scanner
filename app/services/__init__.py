# Business logic
from app.services.llm_service import LLMServicePool
from app.utils.logger import get_logger
from flask import current_app

# llm service pool singleton
llm_service_pool = None


def get_llm_service_pool():
    """Get the LLM service pool singleton instance."""
    global llm_service_pool
    if llm_service_pool is None:
        app_config = current_app.config.get("LLM_CONFIGS")
        api_keys = app_config.get("API_KEY", "").split(",")

        # iterate through api_keys
        service_configs = []
        for api_key in api_keys:
            service_configs.append(
                {
                    "api_url": app_config.get("API_URL"),
                    "api_key": api_key.strip(),
                    "model": app_config.get("API_MODEL"),
                    "timeout": app_config.get("API_TIMEOUT"),
                }
            )
        llm_service_pool = LLMServicePool(service_configs=service_configs)
    return llm_service_pool
