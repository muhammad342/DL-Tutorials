using CSV, DataFrames, Random, Statistics, Printf
Random.seed!(42)

EPOCHS = 3

train = CSV.read("./mnist/mnist_train.csv", DataFrame, header=1)
test = CSV.read("./mnist/mnist_test.csv", DataFrame, header=1)

using Lux, MLUtils, Optimisers, OneHotArrays, Zygote, JLD2
using Flux
rng = Xoshiro(42)


function mlp_loader(data::DataFrame, batch_size_)
    x4dim = reshape(permutedims(Matrix{Float32}(select(data, Not(:label)))), 28, 28, 1, :)
    x4dim = mapslices(x -> reverse(permutedims(x ./ 255), dims=1), x4dim, dims=(1, 2))
    x_flat = reshape(x4dim, 784, :)
    yhot = OneHotArrays.onehotbatch(Vector(data.label), 0:9)
    return MLUtils.DataLoader((x_flat, yhot); batchsize=batch_size_, shuffle=true)
end


function cnn_loader(data::DataFrame; batchsize::Int=512)
    x4dim = reshape(permutedims(Matrix{Float32}(select(data, Not(:label)))), 28, 28, 1, :)
    x4dim = mapslices(x -> reverse(permutedims(x ./ 255), dims=1), x4dim, dims=(1, 2))
    yhot = Flux.onehotbatch(Vector(data.label), 0:9)
    Flux.DataLoader((x4dim, yhot); batchsize, shuffle=true)
end

mlp_model = Lux.Chain(
    Lux.Dense(784 => 120, relu),
    Lux.Dense(120 => 84, relu),
    Lux.Dense(84 => 10),
)

 
cnn_model = Flux.Chain(
    Flux.Conv((5, 5), 1 => 6, relu),
    Flux.MeanPool((2, 2)),
    Flux.Conv((5, 5), 6 => 16, relu),
    Flux.MeanPool((2, 2)),
    Flux.flatten,
    Flux.Dense(256 => 120, relu),
    Flux.Dense(120 => 84, relu),
    Flux.Dense(84 => 10),
)

# Training functions
const lossfn = Lux.CrossEntropyLoss(; logits=Val(true))

function accuracy_mlp(model, ps, st, dataloader)
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

function train_mlp(model, model_name)
    println("\nTraining $(model_name)...")
    start_time = time()
    
    train_dataloader = mlp_loader(train, 512)
    test_dataloader = mlp_loader(test, 10000)
    ps, st = Lux.setup(rng, model)
    vjp = Lux.AutoZygote()
    train_state = Lux.Training.TrainState(model, ps, st, Optimisers.AdamW(lambda=3e-4))
    
    for epoch in 1:EPOCHS
        for (x, y) in train_dataloader
            _, _, _, train_state = Lux.Training.single_train_step!(
                vjp, lossfn, (x, y), train_state,
            )
        end
    end
    
    total_time = time() - start_time
    final_accuracy = accuracy_mlp(model, train_state.parameters, train_state.states, test_dataloader) * 100
    
    x1, y1 = first(train_dataloader)
    memory = @allocated model(x1, ps, st)
    params = sum(length(p) for p in Optimisers.trainables(ps))
    
    return total_time, final_accuracy, memory, params
end

function train_cnn(model, model_name)
    println("Training $(model_name)...")
    start_time = time()
    
    train_data_loader = cnn_loader(train; batchsize=512)
    opt_rule = Flux.AdamW(0.001, (0.9, 0.999), 3e-4)
    opt_state = Flux.setup(opt_rule, model)
    
    for epoch in 1:EPOCHS
        for (x, y) in train_data_loader
            grads = Flux.gradient(m -> Flux.logitcrossentropy(m(x), y), model)
            Flux.update!(opt_state, model, grads[1])
        end
    end
    
    total_time = time() - start_time
    
    (x, y) = only(cnn_loader(test; batchsize=size(test, 1)))
    ŷ = model(x)
    final_accuracy = round(100 * mean(Flux.onecold(ŷ) .== Flux.onecold(y)); digits=2)
    
    x2, y2 = first(train_data_loader)
    memory = @allocated model(x2)
    params = sum(length, Flux.params(model))
    
    return total_time, final_accuracy, memory, params
end


mlp_time, mlp_acc, mlp_mem, mlp_params = train_mlp(mlp_model, "MLP")  
cnn_time, cnn_acc, cnn_mem, cnn_params = train_cnn(cnn_model, "CNN")



println("MLP vs CNN COMPARISON")  

println(@sprintf("%-20s | %-12s | %-12s", "Metric", "MLP", "CNN"))

println(@sprintf("%-20s | %-12.2f | %-12.2f", "Training Time (s)", mlp_time, cnn_time))
println(@sprintf("%-20s | %-12.2f | %-12.2f", "Test Accuracy (%)", mlp_acc, cnn_acc))
println(@sprintf("%-20s | %-12d | %-12d", "Parameters", mlp_params, cnn_params))
