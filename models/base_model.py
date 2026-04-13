import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


class BaseModel:

    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        self.model.eval()

    def generate(self, prompt, max_new_tokens=128,
                 temperature=0.2, top_p=0.9):
        """
        Reliability-first decoding:
        - Greedy (no sampling)
        - Deterministic
        - Suitable for calibration analysis
        """

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # ---- Greedy decoding ----
        with torch.no_grad():
            gen_outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False  # Greedy
            )

        full_sequence = gen_outputs[0]
        prompt_length = inputs["input_ids"].shape[1]

        generated_tokens = full_sequence[prompt_length:]

        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        ).strip()

        # ---- Proper Predictive Entropy Computation ----
        with torch.no_grad():
            outputs = self.model(full_sequence.unsqueeze(0))
            logits = outputs.logits

        logprobs = []
        entropies = []

        for i in range(prompt_length, full_sequence.shape[0]):

            step_logits = logits[0, i - 1].float()

            probs = F.softmax(step_logits, dim=-1)
            log_probs = F.log_softmax(step_logits, dim=-1)

            token_id = full_sequence[i]

            token_logprob = log_probs[token_id].item()
            entropy = -(probs * log_probs).sum().item()

            logprobs.append(token_logprob)
            entropies.append(entropy)

        avg_logprob = float(np.mean(logprobs)) if logprobs else None
        avg_entropy = float(np.mean(entropies)) if entropies else None

        return answer, avg_logprob, avg_entropy