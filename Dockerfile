FROM python:2.7-slim

WORKDIR /tmp
COPY . /tmp

RUN pip --default-timeout=100 install -q -r /tmp/requirements.txt

EXPOSE 3000
EXPOSE 9042

CMD ["python", "app.py"]
