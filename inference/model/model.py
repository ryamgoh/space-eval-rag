class ModelFactory:
    def get_model(model_config: Dict) -> BaseModel
    def create_huggingface_model(config)
    def create_vllm_model(config)
    def create_api_model(config)  # OpenAI, Anthropic, Gemini