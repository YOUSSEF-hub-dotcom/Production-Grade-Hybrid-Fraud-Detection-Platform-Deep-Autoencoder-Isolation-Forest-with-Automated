import logging

def setup_logging():
    # Prevent duplication if the function is used more than once.
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("creditCard_pipeline.log", encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
