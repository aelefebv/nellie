import logging

logger = logging
logger.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s.%(msecs)03d :: %(levelname)s:%(name)s:[%(filename)s:%(lineno)d] :: %(message)s",
    datefmt="%Y-%m-%d | %H:%M:%S",
)
logger.getLogger('xmlschema').setLevel(logger.WARNING)
