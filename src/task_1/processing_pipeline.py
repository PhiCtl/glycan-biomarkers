from src.task_1.processing_helpers import load_data
from typing import List, Callable, Optional

# TODO fix OOP issues and verbose

class PipelineStep():

    def __init__(self, name:str, func: Optional[Callable]=None, **kwargs):
        self.name = name
        self.func = func
        self.func_args = kwargs
    
    def run(self, data=None):
        if 'verbose' in self.func_args.keys():
            print(f"Running {self.name} on data")
        if self.func is None:
            raise ValueError(f"No function assigned to {self.name}")
        return self.func(data, **self.func_args) if data is not None else self.func(**self.func_args)

class Pipeline():

    def __init__(self):
        self.steps = []
    
    def add(self, step: PipelineStep):
        self.steps.append(step)
    
    def add(self, step: List[PipelineStep]):
        self.steps.extend(step)
    
    def run(self, data=None):
        for step in self.steps:
            data = step.run(data)
        return data



        
