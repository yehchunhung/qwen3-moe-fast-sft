import torch
from trl import SFTTrainer

def patched_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    """
    Compute training loss and additionally compute token accuracies
    """
    mode = "train" if self.model.training else "eval"
    (loss, outputs) = super(SFTTrainer, self).compute_loss(
        model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
    )
    if mode == "train":
        # When using padding-free, the attention_mask is not present in the inputs, instead we have cu_seq_lens_q,
        # cu_seq_lens_k, and max_length_k, max_length_q and position_ids.
        if "attention_mask" in inputs:
            num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
        elif "position_ids" in inputs:
            local_num_tokens = torch.tensor(inputs["position_ids"].size(1), device=inputs["position_ids"].device)
            num_tokens_in_batch = self.accelerator.gather_for_metrics(local_num_tokens).sum().item()
        else:
            raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
        self._total_train_tokens += num_tokens_in_batch
    self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

    # Compute token accuracy if we have labels and if the model is not using Liger (no logits)
    if "labels" in inputs and outputs.logits: # PATCH: not self.args.use_liger_kernel
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = inputs["labels"][..., 1:].contiguous()

        # Get predictions
        predictions = shift_logits.argmax(dim=-1)

        # Create mask for non-padding tokens (assuming ignore_index is -100)
        mask = shift_labels != -100

        # Calculate accuracy only on non-padding tokens
        correct_predictions = (predictions == shift_labels) & mask
        total_tokens = mask.sum()
        correct_tokens = correct_predictions.sum()

        # Gather the correct_tokens and total_tokens across all processes
        correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
        total_tokens = self.accelerator.gather_for_metrics(total_tokens)

        # Compute the mean token accuracy and log it
        total_sum = total_tokens.sum()
        accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
        self._metrics[mode]["mean_token_accuracy"].append(accuracy)

    return (loss, outputs) if return_outputs else loss