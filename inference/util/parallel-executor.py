class ParallelExecutor:
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers)
    
    def run_evaluation_batch(model: BaseModel, 
                           prompts: List[str], 
                           batch_size: int) -> Future
    def gather_results(futures: List[Future]) -> List