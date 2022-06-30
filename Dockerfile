FROM nvcr.io/nvidia/tensorflow:22.06-tf2-py3

ARG USERNAME=vorph
ARG USER_UID=1000
ARG USER_GID=1000


RUN groupadd -g $USER_GID -o $USERNAME
RUN useradd -m -u $USER_UID -g $USER_GID -o -s /bin/bash $USERNAME

USER $USERNAME

ENV PATH "$PATH:/home/$USERNAME/.local/bin"

RUN /usr/bin/python -m pip install --upgrade pip
ENV PATH "$PATH:/usr/lib/python3.8/dist-packages"

WORKDIR /home/$USERNAME

COPY requirements.txt .
COPY requirements-dev.txt .

RUN /bin/bash -c "pip install -r requirements.txt --no-cache-dir"
RUN /bin/bash -c "pip install -r requirements-dev.txt --no-cache-dir"

# 5000 pour le monitoring
EXPOSE 5000

# 6006 pour tensorboard
EXPOSE 6006

# 8001 pour mkdocs
EXPOSE 8001
