using Revise

using BenchmarkTools, ProgressMeter
using Statistics, DataFrames, StatsBase, GLM

using CUDA, SparseArrays
using IntervalMDP, IntervalMDP.Data

using Makie, CairoMakie
CairoMakie.activate!(type = "png")


function setup_prism()
    # Ensure PRISM is correctly installed
    cd("prism-4.8.1-linux64-x86/")
    run(`./install.sh`)
    cd("..")
end


function benchmark_bmdp_tool(model, problem; samples=10000, timeout_seconds=5.0)
    @info "bmdp-tool"

    # Write to bmdp-tool format
    base_path = joinpath(@__DIR__, "data", "bmdp-tool")

    bmdp_tool_path = joinpath(base_path, model * ".txt")
    mkpath(dirname(bmdp_tool_path))

    write_bmdp_tool_file(bmdp_tool_path, problem)

    # Extract problem info
    spec = specification(problem)
    strat_mode = strategy_mode(spec) == Maximize ? "maximize" : "minimize"
    sat_mode = satisfaction_mode(spec) == Pessimistic ? "pessimistic" : "optimistic"

    prop = system_property(spec)
    @assert prop isa AbstractReachability
    @assert isfinitetime(prop)
    horizon = time_horizon(prop)

    # Command to run
    # Note: when horizon is not -1, then the 4th arg (eps for convergence) is ignored
    bmdp_tool_cmd = `$(@__DIR__)/bmdp-tool/synthesis $strat_mode $sat_mode $horizon 1e-6 $bmdp_tool_path`

    # Execution statistics
    execution_times = Float64[]
    cumulative_execution_time = 0.0

    # Sample
    for i in 1:samples
        # Run command
        output = read(bmdp_tool_cmd, String)
        
        # Analyze output
        first_line, _ = split(output, '\n'; limit=2)

        microseconds = parse(Int64, first_line)
        seconds = microseconds / 1e6

        # Save statistics
        push!(execution_times, seconds)
        cumulative_execution_time += seconds

        # If more than `timeout_seconds` used, terminate.
        if cumulative_execution_time >= timeout_seconds
            break
        end
    end

    median_seconds = median(execution_times)

    return median_seconds
end


function benchmark_prism(model, problem; samples=10000, timeout_seconds=5.0)
    @info "PRISM"

    # Write to PRISM format
    base_path = joinpath(@__DIR__, "data", "prism")

    prism_path = joinpath(base_path, model)
    mkpath(dirname(prism_path))

    write_prism_file(prism_path, problem)

    # Command to run
    pctl_path = joinpath(base_path, model * ".pctl")
    prop = read(pctl_path, String)

    prism_path = joinpath(base_path, model * ".all")
    prism_cmd = `$(@__DIR__)/prism-4.8.1-linux64-x86/bin/prism -javamaxmem 12g -importmodel $prism_path -pctl $prop`

    # Execution statistics
    execution_times = Float64[]
    cumulative_execution_time = 0.0

    # Sample
    for i in 1:samples
        # Run command
        output = read(prism_cmd, String)

        # Analyze output
        pattern = r"Time for model checking:[^0-9]*(?P<time>[0-9.]*)"
        m = match(pattern, output)

        seconds = parse(Float64, m[:time])

        # Save statistics
        push!(execution_times, seconds)
        cumulative_execution_time += seconds

        # If more than `timeout_seconds` used, terminate.
        if cumulative_execution_time >= timeout_seconds
            break
        end
    end

    median_seconds = median(execution_times)

    return median_seconds
end


function benchmark_intervalmdp_cpu(problem; samples=10000, timeout_seconds=5.0)
    @info "IntervalMDP (CPU)"

    result = @benchmark value_iteration($problem) samples=samples seconds=timeout_seconds evals=1

    median_nanoseconds = time(median(result))
    median_seconds = median_nanoseconds / 1e9

    return median_seconds
end


function benchmark_intervalmdp_gpu(problem; samples=10000, timeout_seconds=5.0)
    if CUDA.functional()
        @info "IntervalMDP (GPU)"

        gpu_problem = IntervalMDP.cu(problem)

        result = @benchmark value_iteration($gpu_problem) samples=samples seconds=timeout_seconds evals=1

        median_nanoseconds = time(median(result))
        median_seconds = median_nanoseconds / 1e9

        return median_seconds
    else
        return missing
    end
end


function benchmark_model(model)
    # Read model
    model_path = joinpath(@__DIR__, "data", model * ".nc")
    spec_path = joinpath(@__DIR__, "data", model * ".json")
    problem = read_intervalmdp_jl(model_path, spec_path)

    # NNZ = Number of Non-Zero elements of the transition matrix
    probabilities = transition_prob(system(problem))
    num_transitions = nnz(upper(probabilities))

    res_bmdp_tool = benchmark_bmdp_tool(model, problem)
    res_prism = benchmark_prism(model, problem)
    res_intervalmdp_cpu = benchmark_intervalmdp_cpu(problem)
    res_intervalmdp_gpu = benchmark_intervalmdp_gpu(problem)

    return (model=model, num_transitions=num_transitions, bmdp_tool=res_bmdp_tool, prism=res_prism, intervalmdp_cpu=res_intervalmdp_cpu, intervalmdp_gpu=res_intervalmdp_gpu)
end

function benchmark()
    # Warn if CUDA is not available
    if !CUDA.functional()
        @warn "CUDA is not available so IntervalMDP (GPU) is skipped"
    end

    # Benchmark models iteratively
    models = [
        "pimdp_0",
        "pimdp_1",
        "pimdp_2",
        "multiObj_robotIMDP",
        "linear/probability_data_5_f_0.9_sigma_[0.01]",
        "linear/probability_data_5_f_0.9_sigma_[0.05]",
        "linear/probability_data_5_f_0.9_sigma_[0.1]",
        "linear/probability_data_5_f_1.05_sigma_[0.01]",
        "linear/probability_data_5_f_1.05_sigma_[0.05]",
        "linear/probability_data_5_f_1.05_sigma_[0.1]",
        "linear/probability_data_50_f_0.9_sigma_[0.01]",
        "linear/probability_data_50_f_0.9_sigma_[0.05]",
        "linear/probability_data_50_f_0.9_sigma_[0.1]",
        "linear/probability_data_50_f_1.05_sigma_[0.01]",
        "linear/probability_data_50_f_1.05_sigma_[0.05]",
        "linear/probability_data_50_f_1.05_sigma_[0.1]",
        "pendulum/probability_data_1_layers_120_sigma_[0.01, 0.01]",
        "pendulum/probability_data_1_layers_120_sigma_[0.05, 0.05]",
        "pendulum/probability_data_1_layers_120_sigma_[0.1, 0.1]",
        "pendulum/probability_data_1_layers_240_sigma_[0.01, 0.01]",
        "pendulum/probability_data_1_layers_240_sigma_[0.05, 0.05]",
        "pendulum/probability_data_1_layers_240_sigma_[0.1, 0.1]",
        "pendulum/probability_data_1_layers_480_sigma_[0.01, 0.01]",
        "pendulum/probability_data_1_layers_480_sigma_[0.05, 0.05]",
        "pendulum/probability_data_1_layers_480_sigma_[0.1, 0.1]",
        "cartpole/probability_data_1_layers_960_sigma_[0.01, 0.01, 0.01, 0.01]",
        "cartpole/probability_data_1_layers_960_sigma_[0.05, 0.05, 0.05, 0.05]",
        "cartpole/probability_data_1_layers_960_sigma_[0.1, 0.1, 0.1, 0.1]",
        "cartpole/probability_data_1_layers_1920_sigma_[0.01, 0.01, 0.01, 0.01]",
        "cartpole/probability_data_1_layers_1920_sigma_[0.05, 0.05, 0.05, 0.05]",
        "cartpole/probability_data_1_layers_1920_sigma_[0.1, 0.1, 0.1, 0.1]",
        "cartpole/probability_data_1_layers_3840_sigma_[0.01, 0.01, 0.01, 0.01]",
        "cartpole/probability_data_1_layers_3840_sigma_[0.05, 0.05, 0.05, 0.05]",
        "cartpole/probability_data_1_layers_3840_sigma_[0.1, 0.1, 0.1, 0.1]",
        "harrier_25920_sigma_[0.05, 0.05, 0.02, 0.01, 0.01, 0.01]"
    ]

    results = []

    @showprogress output=stdout desc="Models: " for model in models
        @info "Benchmarking $model"

        res = benchmark_model(model)
        push!(results, res)
    end

    df = DataFrame(results)

    ENV["DATAFRAMES_ROWS"] = 50

    return df
end

function plot_benchmark(df)
    mkpath("results")

    plot_linear_scale(df)
    plot_loglog_scale(df)
end

function plot_linear_scale(df)
    f = Figure(size = (1000, 500))
    ax = Axis(
        f[1, 1],
        xlabel = "Number of transitions",
        ylabel = "Time (s)", 
        limits = (0, 50000000, 0, 8000),
        xticks = (0:10000000:50000000, ["0E+0", "1E+7", "2E+7", "3E+7", "4E+7", "5E+7"]),
    )

    x = df.num_transitions

    # Scatter plot
    y_bmdp_tool = df.bmdp_tool
    scatter!(ax, x, y_bmdp_tool, color = :red, marker = :rect, markersize = 10, label = "bmdp-tool")

    y_prism = df.prism
    scatter!(ax, x, y_prism, color = :green4, marker = :diamond, markersize = 10, label = "PRISM")

    y_intervalmdp_cpu = df.intervalmdp_cpu
    scatter!(ax, x, y_intervalmdp_cpu, color = :cyan, marker = :utriangle, markersize = 10, label = "IntervalMDP.jl (CPU)")
    
    if !any(ismissing, df.intervalmdp_gpu)
        y_intervalmdp_gpu = df.intervalmdp_gpu
        scatter!(ax, x, y_intervalmdp_gpu, color = :maroon, marker = :circle, markersize = 10, label = "IntervalMDP.jl (GPU)")
    end

    # Linear regression
    xticks = 0:10000000:50000000
    xticks_df = DataFrame(num_transitions = xticks)

    bmdp_tool_ols = lm(@formula(bmdp_tool ~ num_transitions), df)
    y_bmdp_tool_linear = predict(bmdp_tool_ols, xticks_df)
    lines!(ax, xticks, y_bmdp_tool_linear, color = :red, linestyle = :dot, label = "Linear (bmdp-tool)")

    prism_ols = lm(@formula(prism ~ num_transitions), df)
    y_prism_linear = predict(prism_ols, xticks_df)
    lines!(ax, xticks, y_prism_linear, color = :green4, linestyle = :dot, label = "Linear (PRISM)")

    intervalmdp_cpu_ols = lm(@formula(intervalmdp_cpu ~ num_transitions), df)
    y_intervalmdp_cpu_linear = predict(intervalmdp_cpu_ols, xticks_df)
    lines!(ax, xticks, y_intervalmdp_cpu_linear, color = :turquoise3, linestyle = :dot, label = "Linear (IntervalMDP.jl (CPU))")

    if !any(ismissing, df.intervalmdp_gpu)
        intervalmdp_gpu_ols = lm(@formula(intervalmdp_gpu ~ num_transitions), df)
        y_intervalmdp_gpu_linear = predict(intervalmdp_gpu_ols, xticks_df)
        lines!(ax, xticks, y_intervalmdp_gpu_linear, color = :maroon, linestyle = :dot, label = "Linear (IntervalMDP.jl (GPU))")
    end

    f[1, 2] = Legend(f, ax, framevisible = false)

    save("results/linear_scale.png", f)
end

function plot_loglog_scale(df)
    f = Figure(size = (1000, 500))
    ax = Axis(
        f[1, 1],
        xlabel = "Number of transitions",
        ylabel = "Time (s)", 
        limits = (1, 100000000, 1e-4, 1e4),
        xscale=log10,
        yscale=log10,
        xticks = (10 .^ (0:8), ["1E+0", "1E+1", "1E+2", "1E+3", "1E+4", "1E+5", "1E+6", "1E+7", "1E+8"]),
        yticks = (10 .^ (-4.0:4.0), ["1E-4", "1E-3", "1E-2", "1E-1", "1E+0", "1E+1", "1E+2", "1E+3", "1E+4"]),
    )

    x = df.num_transitions

    # Scatter plot
    y_bmdp_tool = df.bmdp_tool
    scatter!(ax, x, y_bmdp_tool, color = :red, marker = :rect, markersize = 10, label = "bmdp-tool")

    y_prism = df.prism
    scatter!(ax, x, y_prism, color = :green, marker = :diamond, markersize = 10, label = "PRISM")

    y_intervalmdp_cpu = df.intervalmdp_cpu
    scatter!(ax, x, y_intervalmdp_cpu, color = :turquoise3, marker = :utriangle, markersize = 10, label = "IntervalMDP.jl (CPU)")

    if !any(ismissing, df.intervalmdp_gpu)
        y_intervalmdp_gpu = df.intervalmdp_gpu
        scatter!(ax, x, y_intervalmdp_gpu, color = :maroon, marker = :circle, markersize = 10, label = "IntervalMDP.jl (GPU)")
    end

    f[1, 2] = Legend(f, ax, framevisible = false)

    save("results/loglog_scale.png", f)
end

setup_prism()
df = benchmark()
show(df)
plot_benchmark(df);