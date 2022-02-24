# usage
# docker build --target=activitysim-dependencies -t activitysim-dependencies .
# docker build --target=activitysim-develop -t activitysim-develop .

################
# dependencies
################
FROM continuumio/anaconda3 AS activitysim-dependencies

WORKDIR /app

# local certificate
ADD cacert.pem /usr/node/cacert.pem
ENV REQUESTS_CA_BUNDLE=/usr/node/cacert.pem

# create the environment
# remove if not needed
COPY conda-environments/activitysim-dev-docker.yml .
RUN conda env create --file=activitysim-dev-docker.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "ASIM-DEV", "/bin/bash", "-c"]

#RUN conda activate myenv
RUN echo "Make sure activitysim dependencies are installed:"
RUN python -c "import openmatrix"

################
# build the latest activitysim develop
# also allow debugging
################

FROM activitysim-dependencies AS activitysim-develop

WORKDIR /app
COPY . ./

RUN pip install -e .

RUN pip install debugpy

# ENTRYPOINT [ "python", "-m", "debugpy", "--listen", "0.0.0.0:5678", "--wait-for-client", "entrypoint.py" ]