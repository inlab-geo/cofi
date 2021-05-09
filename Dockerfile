# syntax=docker/dockerfile:1
#FROM jupyter@sha256:7e599e19278efe674aa1808611b86397bd2d4d9529c0b5f8f8d904da20509826
FROM jupyter/datascience-notebook:python-3.8.6
USER root
RUN sudo apt-get update
RUN sudo apt-get -y install vim less iputils-ping g++ gfortran
RUN sudo adduser jovyan sudo
USER jovyan
RUN conda install -y python-dotenv psycopg2 
COPY --chown=jovyan:users $pwd /home/jovyan
