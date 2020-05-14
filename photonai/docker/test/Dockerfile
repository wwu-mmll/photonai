FROM python:3.7
WORKDIR /test-base
RUN python -m pip install --upgrade pip
RUN python -m pip install --upgrade black
RUN python -m pip install --upgrade coverage
COPY requirements.txt ./requirements.txt
RUN python -m pip  install -r requirements.txt --ignore-installed
#

