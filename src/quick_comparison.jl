
using CSV, DataFrames, Random, Statistics, Printf
Random.seed!(42)

EPOCHS = 3

#===== DATA LOADING =====#
println("📊 Loading data...")
train = CSV.read("./mnist/mnist_train.csv", DataFrame, header=1)
test = CSV.read("./mnist/mnist_test.csv", DataFrame, header=1)

println("🏗️  Setting up models...")

#===== MLP SECTION =====#
using Lux, MLUtils, Optimisers, OneHotArrays, Zygote, JLD2
rng = Xoshiro(42)

# MLP functions
function flatten(x::AbstractArray)
    return reshape(x, :, size(x)[end])
end

function mnistloader(data::DataFrame, batch_size_)
    x4dim = reshape(permutedims(Matrix{Float32}(select(data, Not(:label)))), 28, 28, 1, :)
    x4dim = mapslices(x -> reverse(permutedims(x ./ 255), dims=1), x4dim, dims=(1, 2))
    x4dim = Lux.meanpool((x4dim), (2, 2))
    x4dim = flatten(x4dim)
    yhot = OneHotArrays.onehotbatch(Vector(data.label), 0:9)
    return MLUtils.DataLoader((x4dim, yhot); batchsize=batch_size_, shuffle=true)
end

# MLP Model
model = Lux.Chain(
    Lux.Dense(196 => 14, relu),
    Lux.Dense(14 => 14, relu),
    Lux.Dense(14 => 10),
)

# MLP metrics
const lossfn = Lux.CrossEntropyLoss(; logits=Val(true))

function accuracy(model, ps, st, dataloader)
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

#===== MLP TRAINING =====#
println("\nTraining MLP...")
mlp_start_time = time()

train_dataloader, test_dataloader = mnistloader(train, 512), mnistloader(test, 10000)
ps, st = Lux.setup(rng, model)
vjp = Lux.AutoZygote()
train_state = Lux.Training.TrainState(model, ps, st, Optimisers.AdamW(lambda=3e-4))

for epoch in 1:EPOCHS
    for (x, y) in train_dataloader
        global train_state
        _, _, _, train_state = Lux.Training.single_train_step!(
            vjp, lossfn, (x, y), train_state,
        )
    end
end

mlp_total_time = time() - mlp_start_time
mlp_final_accuracy = accuracy(model, train_state.parameters, train_state.states, test_dataloader) * 100

# MLP memory test
x1, y1 = first(mnistloader(train, 512))
mlp_ps, mlp_st = Lux.setup(rng, model)
mlp_memory = @allocated model(x1, mlp_ps, mlp_st)
mlp_params = sum(length(p) for p in Optimisers.trainables(Lux.setup(rng, model)[1]))

#===== CNN SECTION=====#
using Flux, JLD2

# CNN functions
function loader(data::DataFrame; batchsize::Int=512)
    x4dim = reshape(permutedims(Matrix{Float32}(select(data, Not(:label)))), 28, 28, 1, :)
    x4dim = mapslices(x -> reverse(permutedims(x ./ 255), dims=1), x4dim, dims=(1, 2))
    yhot = Flux.onehotbatch(Vector(data.label), 0:9)
    Flux.DataLoader((x4dim, yhot); batchsize, shuffle=true)
end

# CNN Model
lenet = Flux.Chain(
    Flux.Conv((5, 5), 1 => 6, relu),
    Flux.MeanPool((2, 2)),
    Flux.Conv((5, 5), 6 => 16, relu),
    Flux.MeanPool((2, 2)),
    Flux.flatten,
    Flux.Dense(256 => 120, relu),
    Flux.Dense(120 => 84, relu),
    Flux.Dense(84 => 10),
)

function loss_and_accuracy(model, data)
    (x, y) = only(loader(data; batchsize=size(data, 1)))
    ŷ = model(x)
    loss = Flux.logitcrossentropy(ŷ, y)
    acc = round(100 * mean(Flux.onecold(ŷ) .== Flux.onecold(y)); digits=2)
    return loss, acc
end

#===== CNN TRAINING =====#
println("Training CNN...")
cnn_start_time = time()

train_data_loader = loader(train; batchsize=512)
opt_rule = Flux.AdamW(0.001, (0.9, 0.999), 3e-4)
opt_state = Flux.setup(opt_rule, lenet)

for epoch in 1:EPOCHS
    for (x, y) in train_data_loader
        grads = Flux.gradient(m -> Flux.logitcrossentropy(m(x), y), lenet)
        Flux.update!(opt_state, lenet, grads[1])
    end
end

cnn_total_time = time() - cnn_start_time
_, cnn_final_accuracy = loss_and_accuracy(lenet, test)

x2, y2 = first(loader(train; batchsize=512))
cnn_memory = @allocated lenet(x2)
cnn_params = sum(length, Flux.params(lenet))


println("\nTesting memory usage...")
println("\nRESULTS")


println(@sprintf("%-20s | %-10s | %-10s | %-10s", "Metric", "MLP", "CNN", "Ratio"))
println("-" ^ 60)
println(@sprintf("%-20s | %-10.2f | %-10.2f | %-10.2f", "Training Time (s)", mlp_total_time, cnn_total_time, cnn_total_time/mlp_total_time))
println(@sprintf("%-20s | %-10.2f | %-10.2f | %-10.2f", "Test Accuracy (%)", mlp_final_accuracy, cnn_final_accuracy, cnn_final_accuracy/mlp_final_accuracy))
println(@sprintf("%-20s | %-10.1f | %-10.1f | %-10.2f", "Memory Usage (KB)", mlp_memory/1024, cnn_memory/1024, cnn_memory/mlp_memory))
println(@sprintf("%-20s | %-10d | %-10d | %-10.2f", "Parameters", mlp_params, cnn_params, cnn_params/mlp_params))

if mlp_total_time < cnn_total_time
    println("   Speed: MLP (", round(cnn_total_time/mlp_total_time, digits=2), "x faster)")
else
    println("   Speed: CNN (", round(mlp_total_time/cnn_total_time, digits=2), "x faster)")
end

if mlp_final_accuracy > cnn_final_accuracy
    println("   Accuracy: MLP (+", round(mlp_final_accuracy-cnn_final_accuracy, digits=2), "%)")
else
    println("   Accuracy: CNN (+", round(cnn_final_accuracy-mlp_final_accuracy, digits=2), "%)")
end

println("\nWHY MLP IS FASTER than CNN:")
println("   Input size: 196 features vs 784 (75% reduction via meanpool)")
println("   Parameters: ", mlp_params, " vs ", cnn_params)
println("   MLP has 3 dense layers vs CNN has 2 convolutional layers + 2 dense layers")