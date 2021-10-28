# Dockerfile configuration

## Development environment

The first Docker file is the one used in development, it provides an isolated development environment.

```docker
--8<-- "Dockerfile"
```

This `.json` file is used to permit to work with VSCode inside your development container.
```json
--8<-- ".devcontainer/devcontainer.json"
```

## Production environment

This Dockerfile is the one used for "production" you won't be able to modify the scripts that are inside. This container works as an app :

* you plug-in your volume containing your datas,
* you generate the masks,
* you chose the training configuration,
* you run the training loop
* you gets your model and metrics back.

```docker
--8<-- "Dockerfile.prod"
```
