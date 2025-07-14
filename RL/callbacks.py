from transformers import (
    TrainerCallback,
    TrainingArguments,
    TrainerControl,
    TrainerState,
)

import logging

logger = logging.getLogger(__name__)


class LoggingCallback(TrainerCallback):
    """
    A simple callback for logging training information at specific steps.
    """

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step % args.logging_steps == 0:
            if state.log_history and len(state.log_history) > 0:
                logger.info(
                    f"Step {state.global_step}: Loss = {state.log_history[-1].get('loss', None)}, Learning Rate = {state.log_history[-1].get('learning_rate', None)}"
                )
            else:
                logger.info(
                    f"Step {state.global_step}: No logging information available yet"
                )


def get_callbacks(training_args, model_args, script_args):
    """
    Returns a list of callbacks to be used during training.
    For now, it includes only the LoggingCallback. You can extend this to add more callbacks.
    """
    callbacks = [LoggingCallback()]  # Instantiate our LoggingCallback
    return callbacks
