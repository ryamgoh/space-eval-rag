class ResultAggregator:
    def aggregate_results(results: List[Dict]) -> DataFrame
    def compute_statistics(results: DataFrame) -> Dict
    def save_results(results: DataFrame, format: str = "json")
    def generate_report(results: DataFrame) -> str