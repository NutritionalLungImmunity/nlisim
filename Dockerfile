FROM python:3.7-slim-buster
LABEL maintainer="Aspergillus Developers <aspergillus@mail.computational-biology.org>"

WORKDIR /opt/simulation-framework
COPY nlisim /opt/simulation-framework/nlisim
COPY setup.py setup.cfg /opt/simulation-framework/
COPY .git /opt/simulation-framework/.git
RUN apt update \
    && apt install -y git libgl1 \
    && pip3 install . \
    && apt remove -y git
ENTRYPOINT ["nlisim"]
CMD ["run", "5"]
