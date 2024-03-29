FROM julia:1.10.2-bookworm

# Create workdir at /app
RUN mkdir /app
WORKDIR /app

# Install PRISM
RUN apt update
RUN apt -y install openjdk-17-jre
ADD prism-4.8.1-linux64-x86 /app/prism-4.8.1-linux64-x86/

WORKDIR /app/prism-4.8.1-linux64-x86/
RUN /app/prism-4.8.1-linux64-x86/install.sh
WORKDIR /app

# Add dependecy list and download dependencies
ADD Project.toml /app/
# Disable precompilation, as CUDA is not available during build and CUDA.jl cannot detect the correct runtime
ENV JULIA_PKG_PRECOMPILE_AUTO=0
RUN julia --threads auto --project --eval 'using Pkg; Pkg.instantiate()'

# Add source code
ADD benchmark.jl /app/
ADD bmdp-tool /app/bmdp-tool/
ADD data /app/data/

# Run
CMD ["julia", "--threads", "auto", "--project=.", "benchmark.jl"]