FROM tensorflow/tensorflow:2.3.0-gpu

WORKDIR /

ARG build_number=-1
ENV BUILD_NUMBER=$build_number

RUN apt-get update && apt-get install -y git libsm6 libxext6 libxrender-dev

ARG criterionai_core_path='https://bitbucket.org/criterionai/'
COPY $criterionai_core_path core/

ARG criterionai_app_path=hstest/hstest
COPY $criterionai_app_path $criterionai_app_path/

RUN pip install -e core[cv2-headless,tf2-gpu]
RUN pip install -e $criterionai_app_path

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "hstest.train"]