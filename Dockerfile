FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
WORKDIR /runner
COPY evaluation/candidate_worker.py /runner/candidate_worker.py
USER 65534:65534
ENTRYPOINT ["python", "/runner/candidate_worker.py"]
