FROM python:3.8-slim as builder

# Git is needed to install nlisim (to compute the version)
RUN apt-get update && \
    apt-get install --no-install-recommends --yes \
        git

COPY . /opt/nlisim
# Both PYTHONDONTWRITEBYTECODE and --no-compile are necessary to avoid creating .pyc files
RUN PYTHONDONTWRITEBYTECODE=1 pip install --no-cache-dir --no-compile /opt/nlisim


FROM python:3.8-slim as runtime
LABEL maintainer="Aspergillus Developers <aspergillus@mail.computational-biology.org>"

# Set environment to support Unicode: http://click.pocoo.org/5/python3/#python-3-surrogate-handling
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY --from=builder /usr/local/bin/nlisim /usr/local/bin/nlisim
COPY --from=builder /usr/local/lib/python3.7/site-packages/ /usr/local/lib/python3.7/site-packages/

WORKDIR /opt/nlisim
ENTRYPOINT ["nlisim"]
CMD ["run", "5"]
