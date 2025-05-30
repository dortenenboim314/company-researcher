import functools
import inspect
import logging

class LLMLoggingWrapper:
    """
    A wrapper class for interacting with a language model (LLM).
    It calls the LLM with the specified function and params, and just logs the params, calls, and responses.
    """

    def __init__(self, llm: str):
        self.llm = llm

    def __getattr__(self, name):
        attr = getattr(self.llm, name)

        if inspect.ismethod(attr) or inspect.isfunction(attr):
            @functools.wraps(attr)
            def wrapper(*args, **kwargs):
                logging.debug(f"Called: {name} with args={args}, kwargs={kwargs}")
                result = attr(*args, **kwargs)
                logging.debug(f"Result: {result}")
                return result
            return wrapper

        else:
            logging.info(f"Accessed attribute: {name} -> {attr}")
            return attr