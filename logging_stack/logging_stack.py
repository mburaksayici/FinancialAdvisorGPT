import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("./assets/app.log"), logging.StreamHandler()],
)


logger = logging.getLogger()
