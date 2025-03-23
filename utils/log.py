
import logging
import logging



def get_logger(name='crptmidfreq'):
    """Returns a configured logger with a fixed format."""
    
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger  # Avoid duplicate handlers
    
    # Set log level (INFO and above)
    logger.setLevel(logging.INFO)

    # Define log format: [TIME] MESSAGE
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Attach handler to logger
    logger.addHandler(console_handler)

    return logger
