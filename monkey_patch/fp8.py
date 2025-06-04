"""
Adapted from https://github.com/axolotl-ai-cloud/axolotl/blob/a27b909c5c1c2c561a8d503024b89afcce15226f/src/axolotl/monkeypatch/trainer_accelerator_args.py
"""
import re
import inspect
import logging

import torch
from trl import SFTTrainer

LOG = logging.getLogger(__name__)

ORIGINAL_TRAINER_CODE = """
    # create accelerator object
    self.accelerator = Accelerator(**args)
"""

PATCHED_TRAINER_CODE = """
    if hasattr(self, "additional_accelerator_args"):
        from functools import partial
        from accelerate.utils.ao import filter_linear_layers

        # hardcode the replacement function
        def find_first_last_linear_layers_and_routers(model: torch.nn.Module):
            from accelerate.utils.ao import find_first_last_linear_layers

            # collect first and last linear layers
            first_linear, last_linear = find_first_last_linear_layers(model)

            # collect all router linear layers with names ending with "mlp.gate"
            router_layers = list(
                filter(
                    lambda s: re.search(r".*mlp\.gate$", s),
                    [name for name, _ in model.named_modules()]
                )
            )
            return [first_linear, last_linear, *router_layers]

        layers_to_filter = find_first_last_linear_layers_and_routers(self.model)
        module_filter_func = partial(filter_linear_layers, layers_to_filter=layers_to_filter)
        fp8_kwargs = [
            AORecipeKwargs(
                module_filter_func=module_filter_func
            )
        ] # FP8RecipeKwargs(backend="te")
        additional_args = self.additional_accelerator_args(
            mixed_precision="fp8",
            kwarg_handlers=fp8_kwargs,
            **args
        )
        if additional_args:
            args.update(additional_args)

    # create accelerator object
    self.accelerator = Accelerator(**args)
"""

def find_first_last_linear_layers_and_routers(model: torch.nn.Module):
    from accelerate.utils.ao import find_first_last_linear_layers

    # collect first and last linear layers
    first_linear, last_linear = find_first_last_linear_layers(model)

    # collect all router linear layers with names ending with "mlp.gate"
    router_layers = list(
        filter(
            lambda s: re.search(r".*mlp\.gate$", s),
            [name for name, _ in model.named_modules()]
        )
    )
    return [first_linear, last_linear, *router_layers]


def detab_code(code: str) -> tuple[str, str]:
    try:
        spaces = re.match(r"([\s\t]{1,})", code).group(0)
        code = re.sub(r"^" + spaces, "", code, flags=re.MULTILINE)
    except AttributeError:
        return code, ""
    return code, spaces


def get_create_accelerate_code() -> str:
    training_loop = inspect.getsource(SFTTrainer.create_accelerator_and_postprocess)
    return training_loop


def check_create_accelerate_code_is_patchable() -> bool:
    create_code = get_create_accelerate_code()
    create_code, _ = detab_code(create_code)
    return ORIGINAL_TRAINER_CODE in create_code


def patch_create_accelerate_code_for_fp8():
    """
    monkeypatch create_accelerator_and_postprocess so it checks for additional kwargs
    """
    try:
        create_code = get_create_accelerate_code()
    except OSError:
        return
    SFTTrainer._original_create_accelerator_and_postprocess = (  # pylint: disable=protected-access
        create_code
    )
    create_code, _ = detab_code(create_code)
    if ORIGINAL_TRAINER_CODE not in create_code:
        return

    create_code = create_code.replace(ORIGINAL_TRAINER_CODE, PATCHED_TRAINER_CODE)
    create_code = create_code.replace(
        "def create_accelerator_and_postprocess(",
        "def fixed_create_accelerator_and_postprocess(",
        1,
    )

    # load imports necessary
    import transformers.trainer

    items_to_import = []
    for item in dir(transformers.trainer):
        if item in create_code:
            items_to_import.append(item)

    exec(  # pylint: disable=exec-used  # nosec B102
        "from transformers.trainer import ("
        + ", ".join(x for x in items_to_import)
        + ")",
        globals(),
    )
    exec( # pylint: disable=exec-used  # nosec B102
        "from accelerate.utils import AORecipeKwargs, FP8RecipeKwargs",
        globals()
    )
    exec(create_code, globals())  # pylint: disable=exec-used  # nosec B102
    
    LOG.info("patching create_accelerator_and_postprocess to allow for overrides")
    SFTTrainer.create_accelerator_and_postprocess = fixed_create_accelerator_and_postprocess  # pylint: disable=protected-access  # pylint: disable=undefined-variable  # noqa: F821