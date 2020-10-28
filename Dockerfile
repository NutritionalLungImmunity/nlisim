FROM python:3.7-slim-buster
LABEL maintainer="Aspergillus Developers <aspergillus@mail.computational-biology.org>"

WORKDIR /nlisim
COPY . /nlisim

RUN apt update \
	&& apt install -y git libgl1-mesa-glx \
	&& pip3 install --upgrade pip \
	&& pip install --editable .
ENTRYPOINT ["nlisim"]
CMD ["run", "5"]
