using CSV, DataFrames, Random, Statistics, Printf
Random.seed!(42)

EPOCHS = 3

train = CSV.read("./mnist/mnist_train.csv", DataFrame, header=1)
test = CSV.read("./mnist/mnist_test.csv", DataFrame, header=1)


using Lux, MLUtils, Optimisers, OneHotArrays, Zygote, JLD2
using Flux
rng = Xoshiro(42)

#===== DATA LOADERS =====#

# Lux MLP loader - flattened input
function lux_mlp_loader(data::DataFrame, batch_size_)
    x4dim = reshape(permutedims(Matrix{Float32}(select(data, Not(:label)))), 28, 28, 1, :)
    x4dim = mapslices(x -> reverse(permutedims(x ./ 255), dims=1), x4dim, dims=(1, 2))
    x_flat = reshape(x4dim, 784, :)
    yhot = OneHotArrays.onehotbatch(Vector(data.label), 0:9)
    return MLUtils.DataLoader((x_flat, yhot); batchsize=batch_size_, shuffle=true)
end

# Flux MLP loader - flattened input
function flux_mlp_loader(data::DataFrame; batchsize::Int=512)
    x4dim = reshape(permutedims(Matrix{Float32}(select(data, Not(:label)))), 28, 28, 1, :)
    x4dim = mapslices(x -> reverse(permutedims(x ./ 255), dims=1), x4dim, dims=(1, 2))
    x_flat = reshape(x4dim, 784, :)
    yhot = Flux.onehotbatch(Vector(data.label), 0:9)
    Flux.DataLoader((x_flat, yhot); batchsize, shuffle=true)
end

# CNN loader - 4D input for both frameworks
function cnn_loader_lux(data::DataFrame, batch_size_)
    x4dim = reshape(permutedims(Matrix{Float32}(select(data, Not(:label)))), 28, 28, 1, :)
    x4dim = mapslices(x -> reverse(permutedims(x ./ 255), dims=1), x4dim, dims=(1, 2))
    yhot = OneHotArrays.onehotbatch(Vector(data.label), 0:9)
    return MLUtils.DataLoader((x4dim, yhot); batchsize=batch_size_, shuffle=true)
end

function cnn_loader_flux(data::DataFrame; batchsize::Int=512)
    x4dim = reshape(permutedims(Matrix{Float32}(select(data, Not(:label)))), 28, 28, 1, :)
    x4dim = mapslices(x -> reverse(permutedims(x ./ 255), dims=1), x4dim, dims=(1, 2))
    yhot = Flux.onehotbatch(Vector(data.label), 0:9)
    Flux.DataLoader((x4dim, yhot); batchsize, shuffle=true)
end

#===== MODELS WITH MATCHED PARAMETERS (~44K) =====#

# Lux MLP - reduced first layer to match CNN parameters
lux_mlp = Lux.Chain(
    Lux.Dense(784 => 64, relu),  # Reduced from 120 to 64
    Lux.Dense(64 => 32, relu),   # Reduced accordingly  
    Lux.Dense(32 => 10),
)

# Flux MLP - same architecture as Lux
flux_mlp = Flux.Chain(
    Flux.Dense(784 => 64, relu),
    Flux.Dense(64 => 32, relu),
    Flux.Dense(32 => 10),
)

# Lux CNN - LeNet-5 style
lux_cnn = Lux.Chain(
    Lux.Conv((5, 5), 1 => 6, relu),
    Lux.MeanPool((2, 2)),
    Lux.Conv((5, 5), 6 => 16, relu),
    Lux.MeanPool((2, 2)),
    Lux.FlattenLayer(),
    Lux.Dense(256 => 120, relu),
    Lux.Dense(120 => 84, relu),
    Lux.Dense(84 => 10),
)

# Flux CNN - same architecture as Lux
flux_cnn = Flux.Chain(
    Flux.Conv((5, 5), 1 => 6, relu),
    Flux.MeanPool((2, 2)),
    Flux.Conv((5, 5), 6 => 16, relu),
    Flux.MeanPool((2, 2)),
    Flux.flatten,
    Flux.Dense(256 => 120, relu),
    Flux.Dense(120 => 84, relu),
    Flux.Dense(84 => 10),
)

#===== TRAINING FUNCTIONS =====#

# Lux training functions
const lux_lossfn = Lux.CrossEntropyLoss(; logits=Val(true))

function lux_accuracy(model, ps, st, dataloader)
    total_correct, total = 0, 0
    st = Lux.testmode(st)
    for (x, y) in dataloader
        target_class = OneHotArrays.onecold(y, 0:9)
        predicted_class = OneHotArrays.onecold(Array(first(model(x, ps, st))), 0:9)
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end

function train_lux_mlp(model, model_name)
    println("Training $(model_name)...")
    start_time = time()
    
    train_dataloader = lux_mlp_loader(train, 512)
    test_dataloader = lux_mlp_loader(test, 10000)
    ps, st = Lux.setup(rng, model)
    vjp = Lux.AutoZygote()
    train_state = Lux.Training.TrainState(model, ps, st, Optimisers.AdamW(lambda=3e-4))
    
    for epoch in 1:EPOCHS
        for (x, y) in train_dataloader
            _, _, _, train_state = Lux.Training.single_train_step!(
                vjp, lux_lossfn, (x, y), train_state,
            )
        end
    end
    
    total_time = time() - start_time
    final_accuracy = lux_accuracy(model, train_state.parameters, train_state.states, test_dataloader) * 100
    
    x1, y1 = first(train_dataloader)
    memory = @allocated model(x1, ps, st)
    params = sum(length(p) for p in Optimisers.trainables(ps))
    
    return total_time, final_accuracy, memory, params
end

function train_lux_cnn(model, model_name)
    println("Training $(model_name)...")
    start_time = time()
    
    train_dataloader = cnn_loader_lux(train, 512)
    test_dataloader = cnn_loader_lux(test, 10000)
    ps, st = Lux.setup(rng, model)
    vjp = Lux.AutoZygote()
    train_state = Lux.Training.TrainState(model, ps, st, Optimisers.AdamW(lambda=3e-4))
    
    for epoch in 1:EPOCHS
        for (x, y) in train_dataloader
            _, _, _, train_state = Lux.Training.single_train_step!(
                vjp, lux_lossfn, (x, y), train_state,
            )
        end
    end
    
    total_time = time() - start_time
    final_accuracy = lux_accuracy(model, train_state.parameters, train_state.states, test_dataloader) * 100
    
    x1, y1 = first(train_dataloader)
    memory = @allocated model(x1, ps, st)
    params = sum(length(p) for p in Optimisers.trainables(ps))
    
    return total_time, final_accuracy, memory, params
end

# Flux training functions
function train_flux_mlp(model, model_name)
    println("Training $(model_name)...")
    start_time = time()
    
    train_data_loader = flux_mlp_loader(train; batchsize=512)
    opt_rule = Flux.AdamW(0.001, (0.9, 0.999), 3e-4)
    opt_state = Flux.setup(opt_rule, model)
    
    for epoch in 1:EPOCHS
        for (x, y) in train_data_loader
            grads = Flux.gradient(m -> Flux.logitcrossentropy(m(x), y), model)
            Flux.update!(opt_state, model, grads[1])
        end
    end
    
    total_time = time() - start_time
    
    (x, y) = only(flux_mlp_loader(test; batchsize=size(test, 1)))
    ŷ = model(x)
    final_accuracy = round(100 * mean(Flux.onecold(ŷ) .== Flux.onecold(y)); digits=2)
    
    x2, y2 = first(train_data_loader)
    memory = @allocated model(x2)
    params = sum(length, Flux.params(model))
    
    return total_time, final_accuracy, memory, params
end

function train_flux_cnn(model, model_name)
    println("Training $(model_name)...")
    start_time = time()
    
    train_data_loader = cnn_loader_flux(train; batchsize=512)
    opt_rule = Flux.AdamW(0.001, (0.9, 0.999), 3e-4)
    opt_state = Flux.setup(opt_rule, model)
    
    for epoch in 1:EPOCHS
        for (x, y) in train_data_loader
            grads = Flux.gradient(m -> Flux.logitcrossentropy(m(x), y), model)
            Flux.update!(opt_state, model, grads[1])
        end
    end
    
    total_time = time() - start_time
    
    (x, y) = only(cnn_loader_flux(test; batchsize=size(test, 1)))
    ŷ = model(x)
    final_accuracy = round(100 * mean(Flux.onecold(ŷ) .== Flux.onecold(y)); digits=2)
    
    x2, y2 = first(train_data_loader)
    memory = @allocated model(x2)
    params = sum(length, Flux.params(model))
    
    return total_time, final_accuracy, memory, params
end

#===== RUN ALL COMPARISONS =====#

println("\n🚀 Running comprehensive comparison...")

# Train all models
lux_mlp_time, lux_mlp_acc, lux_mlp_mem, lux_mlp_params = train_lux_mlp(lux_mlp, "Lux MLP")
flux_mlp_time, flux_mlp_acc, flux_mlp_mem, flux_mlp_params = train_flux_mlp(flux_mlp, "Flux MLP")

lux_cnn_time, lux_cnn_acc, lux_cnn_mem, lux_cnn_params = train_lux_cnn(lux_cnn, "Lux CNN")
flux_cnn_time, flux_cnn_acc, flux_cnn_mem, flux_cnn_params = train_flux_cnn(flux_cnn, "Flux CNN")

#===== RESULTS =====#

println("\n" * "="^90)
println("COMPREHENSIVE COMPARISON")
println("="^90)

println(@sprintf("%-20s | %-12s | %-12s | %-12s | %-12s", "Metric", "Lux MLP", "Flux MLP", "Lux CNN", "Flux CNN"))

println(@sprintf("%-20s | %-12.2f | %-12.2f | %-12.2f | %-12.2f", "Training Time (s)", 
    lux_mlp_time, flux_mlp_time, lux_cnn_time, flux_cnn_time))
println(@sprintf("%-20s | %-12.2f | %-12.2f | %-12.2f | %-12.2f", "Test Accuracy (%)", 
    lux_mlp_acc, flux_mlp_acc, lux_cnn_acc, flux_cnn_acc))

