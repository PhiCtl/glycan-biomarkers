from src.processing_helpers import load_data
from typing import List, Callable, Optional

# TODO fix OOP issues and verbose

class PipelineStep():

    def __init__(self, name:str, func: Optional[Callable]=None):
        self.name = name
        self.func = func
    
    def run(self, data=None, verbose=False):
        if verbose:
            print("*"*100, f"\nRunning step {self.name}")
        if self.func is None:
            raise ValueError(f"No function assigned to {self.name}")
        return self.func(data, verbose=verbose)

class PipelineMultiStep(PipelineStep):

    def __init__(self, name: str, func: Optional[List[Callable]] = None):
        super().__init__(name, func=lambda x : x)
        self.funcs = func
    
    def run(self, data=None, verbose=False):
        for f in self.funcs:
            data = f(data, verbose=verbose)
        return data

class DataLoader(PipelineStep):

    def __init__(self, paths, name='Data Loader', func=load_data):
        super().__init__(name=name, func=func)
        self.paths = paths

    def run(self, data=None, verbose=False):
        if verbose:
            print(f"Running {self.name}")
        if verbose:
            print(f"Loading data from files {self.paths}")
        data = self.func(self.paths, verbose=verbose)
        return data


class Pipeline():

    def __init__(self):
        self.steps = []
    
    def add(self, step: PipelineStep):
        self.steps.append(step)
    
    def add(self, step: List[PipelineStep]):
        self.steps.extend(step)
    
    def run(self, data=None, verbose=False):
        for step in self.steps:
            data = step.run(data, verbose=verbose)
        return data



        
