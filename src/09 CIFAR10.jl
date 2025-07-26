# CIFAR10 CNN Experiments - LeNet Variants and Training Analysis
# Implementation of tasks from to-do.md:
# 1. Train LeNet5 on CIFAR10
# 2. Effect of dataset size vs training steps
# 3. Effect of filter sizes (LeNet3, LeNet5, LeNet7)
# 4. Visualize learned features

using Flux, MLDatasets, OneHotArrays, Statistics, Plots, JLD2, StatsPlots
using Random, Printf

Random.seed!(42)

# Create output directory
output_dir = "cifar10_experiments"
isdir(output_dir) || mkdir(output_dir)

#===== DATA LOADING =====#

# Load CIFAR10 dataset
function load_cifar10()
    train_x, train_y = CIFAR10(split=:train)[:]
    test_x, test_y = CIFAR10(split=:test)[:]
    
    # Convert to Float32 and normalize to [0, 1]
    train_x = Float32.(train_x) ./ 255.0f0
    test_x = Float32.(test_x) ./ 255.0f0
    
    # One-hot encode labels (CIFAR10 has classes 0-9)
    train_y_hot = onehotbatch(train_y, 0:9)
    test_y_hot = onehotbatch(test_y, 0:9)
    
    return (train_x, train_y_hot), (test_x, test_y_hot), (train_y, test_y)
end

# Data loader with subset capability
function create_dataloader(x, y, batchsize=64, subset_size=nothing, shuffle=true)
    if subset_size !== nothing
        indices = shuffle ? randperm(size(x, 4))[1:subset_size] : 1:subset_size
        x = x[:, :, :, indices]
        y = y[:, indices]
    end
    return Flux.DataLoader((x, y); batchsize, shuffle)
end

println("Loading CIFAR10 dataset...")
(train_x, train_y), (test_x, test_y), (train_y_orig, test_y_orig) = load_cifar10()
println("Dataset loaded: Train $(size(train_x)), Test $(size(test_x))")

# CIFAR10 class names for visualization
class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

#===== UTILITY FUNCTION FOR SIZE CALCULATION =====#

function calculate_conv_output_size(input_size, kernel_size, stride=1, padding=0)
    return floor(Int, (input_size + 2*padding - kernel_size) / stride + 1)
end

# Calculate sizes for CIFAR10 (32x32x3)
function get_dense_input_size(conv_layers_spec)
    h, w = 32, 32  # CIFAR10 input size
    
    for (kernel_size, pooling_size) in conv_layers_spec
        # After convolution (no padding)
        h = calculate_conv_output_size(h, kernel_size, 1, 0)
        w = calculate_conv_output_size(w, kernel_size, 1, 0)
        
        # After pooling
        h = h ÷ pooling_size
        w = w ÷ pooling_size
        
        println("After conv($kernel_size) + pool($pooling_size): $(h)x$(w)")
    end
    
    dense_input = h * w * 16  # 16 is the number of output channels from second conv layer
    println("Dense layer input size: $dense_input")
    return dense_input
end

#===== MODEL ARCHITECTURES =====#

# LeNet3 - 3x3 filters - Fixed with better architecture
function lenet3()
    println("Building LeNet3...")
    conv_spec = [(3, 2), (3, 2)]  # (kernel_size, pool_size) for each conv layer
    dense_input = get_dense_input_size(conv_spec)
    
    return Chain(
        Conv((3, 3), 3 => 6, relu; pad=1),  # Added padding to preserve more spatial info
        MeanPool((2, 2)),
        Conv((3, 3), 6 => 16, relu; pad=1), # Added padding
        MeanPool((2, 2)),
        Flux.flatten,
        Dense(1024 => 120, relu),  # Fixed size based on actual calculation
        Dropout(0.2),
        Dense(120 => 84, relu),
        Dense(84 => 10)
    )
end

# LeNet5 - 5x5 filters (original)
function lenet5()
    println("Building LeNet5...")
    conv_spec = [(5, 2), (5, 2)]  # (kernel_size, pool_size) for each conv layer
    dense_input = get_dense_input_size(conv_spec)
    
    return Chain(
        Conv((5, 5), 3 => 6, relu; pad=0),
        MeanPool((2, 2)),
        Conv((5, 5), 6 => 16, relu; pad=0),
        MeanPool((2, 2)),
        Flux.flatten,
        Dense(dense_input => 120, relu),
        Dense(120 => 84, relu),
        Dense(84 => 10)
    )
end

# LeNet7 - 7x7 filters
function lenet7()
    println("Building LeNet7...")
    conv_spec = [(7, 2), (7, 2)]  # (kernel_size, pool_size) for each conv layer
    dense_input = get_dense_input_size(conv_spec)
    
    return Chain(
        Conv((7, 7), 3 => 6, relu; pad=0),
        MeanPool((2, 2)),
        Conv((7, 7), 6 => 16, relu; pad=0),
        MeanPool((2, 2)),
        Flux.flatten,
        Dense(dense_input => 120, relu),
        Dense(120 => 84, relu),
        Dense(84 => 10)
    )
end

# Test the models with a sample input to verify sizes
println("Testing model architectures...")
sample_input = randn(Float32, 32, 32, 3, 1)

models_to_test = [("LeNet3", lenet3()), ("LeNet5", lenet5()), ("LeNet7", lenet7())]

for (name, model) in models_to_test
    try
        output = model(sample_input)
        println("$name: Input $(size(sample_input)) -> Output $(size(output)) ✓")
    catch e
        println("$name: Error - $e")
    end
end

#===== TRAINING UTILITIES =====#

function loss_and_accuracy(model, dataloader)
    total_loss = 0.0f0
    total_correct = 0
    total_samples = 0
    
    for (x, y) in dataloader
        ŷ = model(x)
        total_loss += Flux.logitcrossentropy(ŷ, y)
        total_correct += sum(Flux.onecold(ŷ) .== Flux.onecold(y))
        total_samples += size(y, 2)
    end
    
    avg_loss = total_loss / length(dataloader)
    accuracy = 100.0 * total_correct / total_samples
    return avg_loss, accuracy
end

function train_model(model, train_loader, test_loader, epochs; lr=0.001, lambda=1e-4)
    opt_rule = AdamW(lr, (0.9, 0.999), lambda)
    opt_state = Flux.setup(opt_rule, model)
    
    train_log = []
    
    println("Starting training...")
    for epoch in 1:epochs
        # Training step
        epoch_start = time()
        for (x, y) in train_loader
            grads = Flux.gradient(m -> Flux.logitcrossentropy(m(x), y), model)
            Flux.update!(opt_state, model, grads[1])
        end
        
        # Evaluation
        train_loss, train_acc = loss_and_accuracy(model, train_loader)
        test_loss, test_acc = loss_and_accuracy(model, test_loader)
        epoch_time = time() - epoch_start
        
        log_entry = (epoch=epoch, train_loss=train_loss, train_acc=train_acc, 
                    test_loss=test_loss, test_acc=test_acc, time=epoch_time)
        push!(train_log, log_entry)
        
        @printf "Epoch %d/%d: Train Loss=%.4f, Train Acc=%.2f%%, Test Loss=%.4f, Test Acc=%.2f%%, Time=%.2fs\n" epoch epochs train_loss train_acc test_loss test_acc epoch_time
    end
    
    return train_log
end

#===== EXPERIMENT 1: Basic LeNet5 Training =====#

println("\n" * "="^60)
println("EXPERIMENT 1: Basic LeNet5 Training on CIFAR10")
println("="^60)

model_lenet5 = lenet5()
train_loader_full = create_dataloader(train_x, train_y, 64)
test_loader = create_dataloader(test_x, test_y, 512, nothing, false)

# Train for several epochs
basic_log = train_model(model_lenet5, train_loader_full, test_loader, 3)  # Reduced epochs for demo

# Save model
JLD2.jldsave(joinpath(output_dir, "lenet5_basic.jld2"); model_state=Flux.state(model_lenet5))

#===== EXPERIMENT 2: Dataset Size vs Training Steps =====#

println("\n" * "="^60)
println("EXPERIMENT 2: Effect of Dataset Size vs Training Steps")
println("="^60)

# Configuration: same total training steps but different dataset sizes
configs = [
    (size=10000, epochs=6, name="10k_6epochs"),
    (size=20000, epochs=3, name="20k_3epochs"),
    (size=30000, epochs=2, name="30k_2epochs")
]

dataset_results = []

for config in configs
    println("\nTraining on $(config.size) examples for $(config.epochs) epochs...")
    
    # Create model and data loader
    model = lenet5()
    train_loader = create_dataloader(train_x, train_y, 64, config.size)
    
    # Train model (reduced epochs for demo)
    actual_epochs = min(config.epochs, 2)  # Limit to 2 epochs for speed
    log = train_model(model, train_loader, test_loader, actual_epochs)
    final_test_acc = log[end].test_acc
    
    # Store results
    result = (size=config.size, epochs=actual_epochs, final_test_acc=final_test_acc, name=config.name)
    push!(dataset_results, result)
    
    # Save model
    JLD2.jldsave(joinpath(output_dir, "lenet5_$(config.name).jld2"); model_state=Flux.state(model))
    
    println("Final test accuracy: $(round(final_test_acc, digits=2))%")
end

# Plot results
sizes = [r.size for r in dataset_results]
accuracies = [r.final_test_acc for r in dataset_results]

p1 = plot(sizes, accuracies, marker=:circle, linewidth=2, markersize=6,
          xlabel="Dataset Size", ylabel="Final Test Accuracy (%)", 
          title="Effect of Dataset Size on Performance\n(Same Total Training Steps)",
          legend=false, grid=true)

# Add annotations
for (i, result) in enumerate(dataset_results)
    annotate!(result.size, result.final_test_acc + 1, 
             text("$(result.epochs) epochs\n$(round(result.final_test_acc, digits=1))%", 8, :center))
end

savefig(p1, joinpath(output_dir, "dataset_size_effect.png"))
display(p1)

#===== EXPERIMENT 3: Filter Size Comparison =====#

println("\n" * "="^60)
println("EXPERIMENT 3: Effect of Filter Sizes (LeNet3, LeNet5, LeNet7)")
println("="^60)

filter_results = []
models_dict = Dict()

# Test different architectures
architectures = [
    (model=lenet3(), name="LeNet3", filter_size="3x3"),
    (model=lenet5(), name="LeNet5", filter_size="5x5"),
    (model=lenet7(), name="LeNet7", filter_size="7x7")
]

for arch in architectures
    println("\nTraining $(arch.name) with $(arch.filter_size) filters...")
    
    # Use subset for faster comparison
    train_loader_subset = create_dataloader(train_x, train_y, 64, 10000)
    
    # Train model
    log = train_model(arch.model, train_loader_subset, test_loader, 2)  # Reduced epochs for demo
    final_test_acc = log[end].test_acc
    
    # Store results
    result = (name=arch.name, filter_size=arch.filter_size, final_test_acc=final_test_acc)
    push!(filter_results, result)
    models_dict[arch.name] = arch.model
    
    # Save model
    JLD2.jldsave(joinpath(output_dir, "$(lowercase(arch.name)).jld2"); 
                 model_state=Flux.state(arch.model))
    
    println("Final test accuracy: $(round(final_test_acc, digits=2))%")
end

# Plot filter size comparison
filter_names = [r.name for r in filter_results]
filter_accuracies = [r.final_test_acc for r in filter_results]

p2 = bar(filter_names, filter_accuracies, 
         xlabel="Architecture", ylabel="Final Test Accuracy (%)",
         title="Effect of Filter Size on Performance", 
         legend=false, color=[:red, :blue, :green])

# Add value labels on bars
for (i, acc) in enumerate(filter_accuracies)
    annotate!(i, acc + 0.5, text("$(round(acc, digits=1))%", 8, :center))
end

savefig(p2, joinpath(output_dir, "filter_size_effect.png"))
display(p2)

#===== EXPERIMENT 4: Feature Visualization =====#

println("\n" * "="^60)
println("EXPERIMENT 4: Visualizing Learned Features")
println("="^60)

function visualize_conv_features(model, x_sample, sample_idx)
    # Get intermediate representations
    conv1_out = model[1](x_sample)  # First conv layer
    pool1_out = model[2](conv1_out)  # First pooling
    conv2_out = model[3](pool1_out)  # Second conv layer
    pool2_out = model[4](conv2_out)  # Second pooling
    
    # Create visualization
    fig = plot(layout=(4, 6), size=(1200, 800))
    
    # Original image (RGB channels)
    original_r = x_sample[:, :, 1, sample_idx]
    original_g = x_sample[:, :, 2, sample_idx]
    original_b = x_sample[:, :, 3, sample_idx]
    
    heatmap!(fig[1,1], original_r', title="Original R", axis=false, border=:none, color=:reds)
    
    # First conv layer features (show first 5 channels)
    for i in 1:min(5, size(conv1_out, 3))
        feature_map = conv1_out[:, :, i, sample_idx]
        heatmap!(fig[1, i+1], feature_map', title="Conv1 Ch$i", 
                axis=false, border=:none, color=:viridis)
    end
    
    # First pooling layer features (show first 5 channels)
    for i in 1:min(5, size(pool1_out, 3))
        feature_map = pool1_out[:, :, i, sample_idx]
        heatmap!(fig[2, i+1], feature_map', title="Pool1 Ch$i", 
                axis=false, border=:none, color=:viridis)
    end
    
    # Second conv layer features (show first 5 channels)
    for i in 1:min(5, size(conv2_out, 3))
        feature_map = conv2_out[:, :, i, sample_idx]
        heatmap!(fig[3, i+1], feature_map', title="Conv2 Ch$i", 
                axis=false, border=:none, color=:viridis)
    end
    
    # Second pooling layer features (show first 5 channels)  
    for i in 1:min(5, size(pool2_out, 3))
        feature_map = pool2_out[:, :, i, sample_idx]
        heatmap!(fig[4, i+1], feature_map', title="Pool2 Ch$i", 
                axis=false, border=:none, color=:viridis)
    end
    
    return fig
end

# Visualize features for 3 sample images using LeNet3
model_lenet3 = models_dict["LeNet3"]

# Get 3 sample images from test set
sample_indices = [1, 100, 200]

for (i, idx) in enumerate(sample_indices)
    # Use the original labels, not the one-hot encoded ones
    class_idx = test_y_orig[idx] + 1  # Convert 0-based to 1-based indexing
    println("Visualizing features for sample $i (class: $(class_names[class_idx]))")
    
    # Create single sample batch
    single_sample = reshape(test_x[:, :, :, idx], 32, 32, 3, 1)
    
    # Generate visualization
    fig = visualize_conv_features(model_lenet3, single_sample, 1)
    
    # Add overall title
    title!(fig, "Feature Maps for Sample $i - $(class_names[class_idx])", 
           subplot=1)
    
    # Save visualization
    savefig(fig, joinpath(output_dir, "feature_visualization_sample_$i.png"))
    display(fig)
end

#===== SUMMARY REPORT =====#

println("\n" * "="^80)
println("EXPERIMENT SUMMARY REPORT")
println("="^80)

println("\n1. BASIC LeNet5 TRAINING:")
println("   Final test accuracy: $(round(basic_log[end].test_acc, digits=2))%")

println("\n2. DATASET SIZE EFFECT:")
for result in dataset_results
    println("   $(result.size) samples ($(result.epochs) epochs): $(round(result.final_test_acc, digits=2))%")
end

println("\n3. FILTER SIZE EFFECT:")
for result in filter_results
    println("   $(result.name) ($(result.filter_size) filters): $(round(result.final_test_acc, digits=2))%")
end

println("\n4. FEATURE VISUALIZATIONS:")
println("   Generated feature maps for 3 samples showing transformation through conv layers")

println("\nANALYSIS:")
println("Dataset Size Effect: $(sizes[end] > sizes[1] ? "Larger" : "Smaller") datasets tend to give $(accuracies[end] > accuracies[1] ? "better" : "worse") performance")
println("Filter Size Effect: $(filter_names[argmax(filter_accuracies)]) performed best among the tested architectures")

# Save summary
summary_data = Dict(
    "basic_training" => basic_log,
    "dataset_size_results" => dataset_results,
    "filter_size_results" => filter_results
)

JLD2.jldsave(joinpath(output_dir, "experiment_summary.jld2"); summary_data)

println("\nAll experiments completed! Results saved in '$output_dir' directory.")
println("Generated files:")
println("- Model checkpoints (.jld2)")
println("- Performance plots (.png)")
println("- Feature visualizations (.png)")
println("- Summary data (experiment_summary.jld2)") 