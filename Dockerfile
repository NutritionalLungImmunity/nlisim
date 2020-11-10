FROM python:3.7-slim-buster
LABEL maintainer="Aspergillus Developers <aspergillus@mail.computational-biology.org>"

WORKDIR /opt/nlisim
COPY nlisim /opt/nlisim/nlisim
COPY setup.py setup.cfg /opt/nlisim/
COPY .git /opt/nlisim/.git
RUN apt update \
    && apt install -y git libgl1 \
    && pip3 install . \
    && apt remove -y git
ENTRYPOINT ["nlisim"]
CMD ["run", "5"]
