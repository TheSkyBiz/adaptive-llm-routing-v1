from models.base_model import BaseModel


class SLMModel(BaseModel):

    def __init__(self):
        super().__init__(
            model_name="Qwen/Qwen2.5-1.5B-Instruct"
        )