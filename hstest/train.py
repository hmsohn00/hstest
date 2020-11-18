import logging
from .hstest import Hstest

log = logging.getLogger(__name__)


if __name__ == "__main__":
    job = Hstest.init(log_level=logging.DEBUG)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    log.info("Start Training")
    output_path = job.train()

    # Complete job
    job.upload_job_output()
    job.complete_job(output_path)
    log.info("Training Done!")