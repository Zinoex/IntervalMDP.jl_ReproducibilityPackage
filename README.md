# Reproducbility package for "IntervalMDP.jl: Accelerated Value Iteration for Interval Markov Decision Processes"
The source code for the Julia package can be found at [https://github.com/Zinoex/IntervalMDP.jl](https://github.com/Zinoex/IntervalMDP.jl). The source code for the reproducibility package of the accompanying paper can be found at [https://github.com/Zinoex/IntervalMDP.jl_ReproducibilityPackage](https://github.com/Zinoex/IntervalMDP.jl_ReproducibilityPackage). The package can be installed from the Julia General package repository using the following commands in the Julia REPL:

```julia
using Pkg
Pkg.add("IntervalMDP")
```


## Hardware requirements
- RAM: at least 16GB
- GPU: NVIDIA GPU with CUDA support (tested on NVIDIA GeForce GTX 1060 6GB), at least 6GB VRAM

## Running instructions
We have prepared a Docker image to build and run the experiments including the baseline. To build and run the experiments, excluding our GPU implementation (see ["Running with NVIDIA GPU"](#running-with-nvidia-gpu) to include the GPU implementation in the benchmark), please clone or copy the source code of [the reproducibility package](https://github.com/Zinoex/IntervalMDP.jl_ReproducibilityPackage) and execute the following two commands from a terminal in the root of the source directory:

```sh
docker build -t intervalmdp-reproducibility-package .
docker run --mount type=bind,source="$(pwd)"/results,target=/app/results intervalmdp-reproducibility-package
```

Note: It may be necessary to execute Docker commands with `sudo` (default) unless the system has been configured to enable sudo-free Docker.

The experiments will run for 8-10 hours, depending on the hardware. The resulting figures will be saved in the `results` directory in the root of the source directory. 

### Running with NVIDIA GPU
To run the GPU accelerated code, the container must be run on a host computer with an NVIDIA GPU that has [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/1.14.5/install-guide.html) installed and configured for Docker (see more details about enabling CUDA for Docker at [Saturn Cloud's "How to Use GPUs from a Docker Container"](https://saturncloud.io/blog/how-to-use-gpu-from-a-docker-container-a-guide-for-data-scientists-and-software-engineers/)). If not running on a machine with an NVIDIA GPU and with the NVIDIA container toolkit correctly installed and configured, a warning is emitted and the GPU benchmark is skipped .

```sh
docker build -t intervalmdp-reproducibility-package .
docker run --gpus all --mount type=bind,source="$(pwd)"/results,target=/app/results intervalmdp-reproducibility-package
```

### Alternative: direct execution
As an alternative to the Docker image, the experiment may be run directly through Julia (version >=1.9). Again, this requires a computer with an NVIDIA GPU. Futhermore, it must run Linux (for bmdp-tool, tested on Manjaro 5.10.211-1 and Debian 12.5) and have Java runtime (version >=9, tested with OpenJDK 17) installed (for PRISM). With those three requirements, the experiment can be run with the following command from the root of the source directory:

```sh
julia --threads auto --project=. --eval 'using Pkg; Pkg.instantiate(); include("benchmark.jl")'
```

## Coverage
Figures 1, 2, and 3 are generated in a drawing program and thus not included in the reproducibility package. Figure 4 is generated by the experiment, running the instructions above. Since the experiments are measurements of execution time (only model checking, not data loading time), the numerical values will not match the ones in the paper exactly. However, the trends and conclusions should remain the same.


## Documentation & testing
You may find basic instructions for installation and usage of the package in the README.md of the package repository, [README.md](https://github.com/Zinoex/IntervalMDP.jl/blob/main/README.md). Extended documentation including usage, theory, data formats, algorithmic choices, and API reference can be found at [https://baymler.com/IntervalMDP.jl/](https://baymler.com/IntervalMDP.jl/dev/). The code has been tested using unit and integration testing (via Julia's testing framework) - the code for testing can be found at [IntervalMDP.jl/test/](https://github.com/Zinoex/IntervalMDP.jl/tree/main/test). Note that while Codecov reports 75% coverage (linked with badge in the README.md of the package), the true coverage is higher. This is due to a lack of GPU-enabled Continuous Integration (CI) nodes for Github Actions, i.e. the GPU code cannot be automatically tested as part of the CI pipeline but has been tested locally with access to an NVIDIA GPU.
