class ConfigManager:
    def load_yaml(config_path: str) -> Dict
    def validate_config(config: Dict) -> bool
    def merge_defaults(config: Dict) -> Dict