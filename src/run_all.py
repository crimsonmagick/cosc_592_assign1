import multiprocessing

import tensorflow as tf
import importlib
import inspect
import pkgutil
from concurrent.futures import ProcessPoolExecutor

import src.cifar_tests
from src.cifar_tests.model_runner import ModelRunner


def get_classes_from_subpackage(package):
    classes = []
    package_name = package.__name__
    
    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        full_module_name = f"{package_name}.{module_name}"
        module = importlib.import_module(full_module_name)
        
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == full_module_name:
                classes.append(obj)
    
    return classes

def run_task(test_class):
    test_class('../report').run_test()

def run_classes(tests):
    with ProcessPoolExecutor(max_workers=12) as executor:
        running_tasks = [executor.submit(run_task, test) for test in tests]
        for running_task in running_tasks:
            running_task.result()


if __name__ == "__main__":
    all_classes = list(filter(lambda to_check: to_check != ModelRunner and (issubclass(to_check, ModelRunner)),
                              get_classes_from_subpackage(src.cifar_tests)))
    run_classes(all_classes)
