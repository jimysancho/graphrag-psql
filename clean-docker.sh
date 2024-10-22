#!/bin/bash

docker compose down 2> /dev/null
docker volume rm lightrag-postgresql_postgres
docker image ls | awk '/lightrag/ {print $3}' | xargs docker rmi
docker compose up -d