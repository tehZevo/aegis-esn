FROM python:3

WORKDIR /app

RUN mkdir -p /app/models

COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

EXPOSE 80

CMD [ "python", "-u", "main.py" ]
