import os
from pathlib import Path

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.rails.llm.options import (
    RailStatus,
    RailType,
)

from app.core.config import get_settings
from app.core.exceptions import GuardrailError
from app.core.logging import get_logger


logger = get_logger(__name__)


class GuardrailService:
    """
    NeMo Guardrails service.

    Flow:

        User input
            ↓
        Input Guardrail
            ↓
        RAG Pipeline
            ↓
        Generated Answer
            ↓
        Output Guardrail
            ↓
        Final Answer
    """

    def __init__(self) -> None:
        settings = get_settings()

        # -------------------------------------------------
        # IMPORTANT
        # -------------------------------------------------
        # Our application settings already load the key
        # from .env.
        #
        # NeMo Guardrails creates its own OpenAI client,
        # which expects OPENAI_API_KEY to be available as
        # an environment variable.
        #
        # We expose the already-loaded application key to
        # NeMo without hardcoding it in config.yml.
        # -------------------------------------------------

        if not settings.openai_api_key:
            raise GuardrailError(
                "OPENAI_API_KEY is not configured."
            )

        os.environ["OPENAI_API_KEY"] = (
            settings.openai_api_key
        )

        config_path = Path(
            __file__
        ).resolve().parent

        try:
            self.config = (
                RailsConfig.from_path(
                    str(config_path)
                )
            )

            self.rails = LLMRails(
                self.config
            )

            logger.info(
                "NeMo Guardrails initialized"
            )

        except Exception as exc:
            logger.exception(
                "Unable to initialize "
                "NeMo Guardrails"
            )

            raise GuardrailError(
                "Unable to initialize guardrails."
            ) from exc

    # =====================================================
    # INPUT GUARDRAIL
    # =====================================================

    async def check_input(
        self,
        user_input: str,
    ) -> bool:
        """
        Return True if user input is allowed.

        Return False if the input guardrail blocks it.
        """

        user_input = user_input.strip()

        if not user_input:
            return False

        try:
            result = (
                await self.rails.check_async(
                    messages=[
                        {
                            "role": "user",
                            "content": user_input,
                        }
                    ],
                    rail_types=[
                        RailType.INPUT
                    ],
                )
            )

            allowed = (
                result.status
                != RailStatus.BLOCKED
            )

            logger.info(
                "Input guardrail result "
                "allowed=%s status=%s",
                allowed,
                result.status,
            )

            return allowed

        except Exception as exc:
            logger.exception(
                "Input guardrail check failed"
            )

            raise GuardrailError(
                "Input guardrail check failed."
            ) from exc

    # =====================================================
    # OUTPUT GUARDRAIL
    # =====================================================

    async def check_output(
        self,
        assistant_output: str,
    ) -> bool:
        """
        Return True if assistant output is allowed.

        Return False if the output guardrail blocks it.
        """

        assistant_output = (
            assistant_output.strip()
        )

        if not assistant_output:
            return False

        try:
            result = (
                await self.rails.check_async(
                    messages=[
                        {
                            "role": "assistant",
                            "content":
                                assistant_output,
                        }
                    ],
                    rail_types=[
                        RailType.OUTPUT
                    ],
                )
            )

            allowed = (
                result.status
                != RailStatus.BLOCKED
            )

            logger.info(
                "Output guardrail result "
                "allowed=%s status=%s",
                allowed,
                result.status,
            )

            return allowed

        except Exception as exc:
            logger.exception(
                "Output guardrail check failed"
            )

            raise GuardrailError(
                "Output guardrail check failed."
            ) from exc