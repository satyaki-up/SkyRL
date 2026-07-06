from typing import TYPE_CHECKING

from .base import GeneratorInput, GeneratorInterface, GeneratorOutput

if TYPE_CHECKING:
    from .skyrl_gym_generator import SkyRLGymGenerator
    from .skyrl_vlm_generator import SkyRLVLMGymGenerator

__all__ = [
    "GeneratorInterface",
    "GeneratorInput",
    "GeneratorOutput",
    "SkyRLGymGenerator",
    "SkyRLVLMGymGenerator",
]


def __getattr__(name: str):
    if name == "SkyRLGymGenerator":
        from .skyrl_gym_generator import SkyRLGymGenerator

        return SkyRLGymGenerator
    if name == "SkyRLVLMGymGenerator":
        from .skyrl_vlm_generator import SkyRLVLMGymGenerator

        return SkyRLVLMGymGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
