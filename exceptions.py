class ErrorReadingData(Exception):
    """Exception raised for errors encountered during data reading."""
    def __init__(self, message=None, file_path=None, file_type=None):
        self.message = message
        self.file_path = file_path
        self.file_type = file_type

    def __str__(self):
        if self.message and self.file_path and self.file_type:
            return f"Error Reading {self.file_type} file '{self.file_path}': {self.message}"
        elif self.message and self.file_path:
            return f"Error Reading file '{self.file_path}': {self.message}"
        elif self.message:
            return f"Error: {self.message}"
        else:
            return "ErrorReadingData has been raised"

class NoModelGiven(Exception):
    """Exception raised when no model is specified."""
    def __init__(self, message="No model specified"):
        self.message = message

    def __str__(self):
        return f"NoModelGiven: {self.message}"

class ModelNotSaved(Exception):
    """Exception raised when a model fails to save."""
    def __init__(self, model_name):
        self.model_name = model_name

    def __str__(self):
        return f"Failed to save model '{self.model_name}'"

class ModelNotLoaded(Exception):
    """Exception raised when a model fails to load."""
    def __init__(self, model_name):
        self.model_name = model_name

    def __str__(self):
        return f"Failed to load model '{self.model_name}'"

class TokenizationError(Exception):
    """Exception raised when tokenization fails."""
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return f"TokenizationError: {self.message}"

class ModelTrainingError(Exception):
    """Exception raised when model training encounters an error."""
    def __init__(self, message):
        self.message = message

    def __str__(self):
        return f"ModelTrainingError: {self.message}"
