from inference.config.manager import ConfigManager


class LLMEvaluator:
    def __init__(self, config_path: str):
        self.config = ConfigManager.load_yaml(config_path)
        self.models = self._load_models()
        self.tasks = self._load_tasks()
        
    def run(self):
        with ParallelExecutor(self.config['parallel']['max_workers']) as executor:
            futures = []
            for model in self.models:
                for task in self.tasks:
                    future = executor.submit(self._evaluate_single, model, task)
                    futures.append(future)
            
            results = executor.gather_results(futures)
            aggregated = ResultAggregator.aggregate_results(results)
            return aggregated
    
    def _evaluate_single(self, model: BaseModel, task: Dict) -> Dict:
        # Load and prepare dataset
        dataset = TaskProcessor.load_dataset(task['dataset'])
        prompts = TaskProcessor.apply_template(dataset, task['prompt_template'])
        
        # Apply RAG if enabled
        if task.get('enable_rag', False):
            rag = RAGManager(task['rag_config']['vector_store'])
            prompts = [rag.augment_prompt(p) for p in prompts]
        
        # Run inference
        predictions = model.batch_generate(prompts, self.config['batch_size'])
        
        # Compute metrics
        metrics = TaskProcessor.compute_metrics(
            predictions, 
            dataset[task['input_mappings'].values()], 
            task['metrics']
        )
        
        return {
            'model': model.name,
            'task': task['name'],
            'metrics': metrics,
            'predictions': predictions[:10]  # Sample for inspection
        }