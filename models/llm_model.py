from models.base_model import BaseModel


class LLMModel(BaseModel):

    def __init__(self):
        super().__init__(
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"
        )