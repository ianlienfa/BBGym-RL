# Get the base Ubuntu image from Docker Hub
FROM ubuntu:20.04

# Update apps on the base image
RUN apt-get -y update && apt-get install -y

# Install the gcc compiler
RUN apt-get -y install build-essential

# Install gcc manual pages
RUN apt-get -y install manpages-dev

# Copy the current folder which contains C++ source code to the Docker image under /usr/src
COPY . /usr/src/dockerBB_smart

# Specify the working directory
WORKDIR /usr/src/dockerBB_smart

# Use Clang to compile the Test.cpp source file
CMD echo "docker hello world"

# Run the output program from the previous step
CMD make 