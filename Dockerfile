FROM python:3.10-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

WORKDIR /app

COPY pytdml-1.2.0-py3-none-any.whl .

RUN pip install --no-cache-dir pytdml-1.2.0-py3-none-any.whl[io]

RUN rm pytdml-1.2.0-py3-none-any.whl

RUN python -c "import pytdml; print(f'PyTDLM version: {pytdml.__version__}')"

CMD ["python"]
