"""Wrappers used by the released image-policy configurations."""
from .multi_step import MultiStep
from .multi_step_full import MultiStepFull
from .robomimic_image import RobomimicImageWrapper

wrapper_dict = {
    "multi_step": MultiStep,
    "multi_step_full": MultiStepFull,
    "robomimic_image": RobomimicImageWrapper,
}
