class TaskProcessor:
    def load_dataset(dataset_config: Dict) -> Dataset
    def apply_template(dataset: Dataset, template: str, mappings: Dict) -> List[str]
    def compute_metrics(predictions: List[str], references: List, metrics: List[str]) -> Dict