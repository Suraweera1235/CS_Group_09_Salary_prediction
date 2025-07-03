# Use Miniconda as base image
FROM continuumio/miniconda3

WORKDIR /app

COPY requirements.txt .

# Create environment and install Python packages
RUN conda create -n ml-env python=3.11 -y && \
    conda run -n ml-env pip install --no-cache-dir -r requirements.txt && \
    conda run -n ml-env pip install ipykernel jupyter && \
    conda run -n ml-env python -m ipykernel install --user --name=ml-env --display-name "Python (ml-env)"

ENV CONDA_DEFAULT_ENV=ml-env
ENV PATH /opt/conda/envs/ml-env/bin:$PATH

COPY . .

EXPOSE 8000

CMD ["python", "src/train.py"]

