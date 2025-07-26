# ## Simple linear regression

# `MLJ` essentially serves as a unified path to many existing Julia packages each of which provides their own functionalities and models, with their own conventions.
#
# The simple linear regression demonstrates this.
# Several packages offer it (beyond just using the backslash operator): here we will use `MLJLinearModels` but we could also have used `GLM`, `ScikitLearn` etc.
#
# To load the model from a given package use `@load ModelName pkg=PackageName`

using MLJ
using MLJModels
models()

filter(model) = model.is_pure_julia && model.is_supervised && model.prediction_type == :probabilistic
models(filter)
models("XGB")
measures("F1")

# Linear regression

LR = @load LinearRegressor pkg = MLJLinearModels verbosity=0

# Note: in order to be able to load this, you **must** have the relevant package in your environment, if you don't, you can always add it (``using Pkg; Pkg.add("MLJLinearModels")``).
#
# Let's load the _boston_ data set

import RDatasets: dataset
import DataFrames: describe, select, Not, rename!
data = dataset("MASS", "Boston")
println(first(data, 3))

# Let's get a feel for the data

@show describe(data)

# So there's no missing value and most variables are encoded as floating point numbers.
# In MLJ it's important to specify the interpretation of the features (should it be considered as a Continuous feature, as a Count, ...?), see also [this tutorial section](/getting-started/choosing-a-model/#data_and_its_interpretation) on scientific types.
#
# Here we will just interpret the integer features as continuous as we will just use a basic linear regression:

data = coerce(data, autotype(data, :discrete_to_continuous))

# Let's also extract the target variable (`MedV`):

y = data.MedV
X = select(data, Not(:MedV))

mdls = models(matching(X, y))

# Let's declare a simple multivariate linear regression model:

model = LR()

# First let's do a very simple univariate regression, in order to fit it on the data, we need to wrap it in a _machine_ which, in MLJ, is the composition of a model and data to apply the model on:

X_uni = select(X, :LStat) # only a single feature
mach_uni = machine(model, X_uni, y)
fit!(mach_uni)
ŷ = MLJ.predict(mach_uni, X_uni)
round(rsquared(ŷ, y), sigdigits=4)
# You can then retrieve the  fitted parameters using `fitted_params`:

fp = fitted_params(mach_uni)
@show fp.coefs
@show fp.intercept

# You can also visualise this

using Plots

plot(X.LStat, y, seriestype=:scatter, markershape=:circle, legend=false, size=(800, 600), xlabel="LStat")

#  MLJ.predict(mach_uni, Xnew) to predict from a fitted model
Xnew = (LStat=collect(range(extrema(X.LStat)..., length=100)),)
plot!(Xnew.LStat, MLJ.predict(mach_uni, Xnew), linewidth=3, color=:orange)


# The  multivariate linear regression case is very similar

mach = machine(model, X, y)
fit!(mach)

fp = fitted_params(mach)
coefs = fp.coefs
intercept = fp.intercept
for (name, val) in coefs
    println("$(rpad(name, 8)):  $(round(val, sigdigits=3))")
end
println("Intercept: $(round(intercept, sigdigits=3))")

# You can use the `machine` in order to _predict_ values as well and, for instance, compute the root mean squared error:

ŷ = MLJ.predict(mach, X)
round(rsquared(ŷ, y), sigdigits=4)

# Let's see what the residuals look like

res = ŷ .- y
begin
    plot(res, line=:stem, linewidth=1, marker=:circle, legend=false, size=((800, 600)))
    hline!([0], linewidth=2, color=:red)    # add a horizontal line at x=0
end
mean(y)

# Maybe that a histogram is more appropriate here

histogram(res, normalize=true, size=(800, 600), label="residual")


# ## Interaction and transformation
#
# Let's say we want to also consider an interaction term of `lstat` and `age` taken together.
# To do this, just create a new dataframe with an additional column corresponding to the interaction term:

X2 = hcat(X, X.LStat .* X.Age)

# So here we have a DataFrame with one extra column corresponding to the elementwise products between `:LStat` and `Age`.
# DataFrame gives this a default name (`:x1`) which we can change:

rename!(X2, :x1 => :interaction)

# Ok cool, now let's try the linear regression again

mach = machine(model, X2, y)
fit!(mach)
ŷ = MLJ.predict(mach, X2)
round(rsquared(ŷ, y), sigdigits=4)

# We get slightly better results but nothing spectacular.
#
# Let's consider regressing the target variable on `lstat` and `lstat^2`; again:
using DataFrames
X3 = DataFrame(hcat(X.LStat, X.LStat .^ 2), [:LStat, :LStat2])
mach = machine(model, X3, y)
fit!(mach)
ŷ = MLJ.predict(mach, X3)
round(rsquared(ŷ, y), sigdigits=4)

# fitting y=mx+c to LStat^2 is the same as fitting y=mx2+c to LStat => Polynomial regression

# which again, we can visualise:

Xnew = (LStat=Xnew.LStat, LStat2=Xnew.LStat .^ 2)

plot(X.LStat, y, seriestype=:scatter, markershape=:circle, legend=false, size=(800, 600))
plot!(Xnew.LStat, MLJ.predict(mach, Xnew), linewidth=3, color=:orange)

# TODO HW : Find the best model by feature selection; best model means highest R²

# ## SOLUTION: Feature Selection for Best Model (Highest R²)

using Statistics, StatsBase, Combinatorics

# First, let's establish baseline performance with all features
println("=== BASELINE MODEL (All Features) ===")
mach_all = machine(model, X, y)
fit!(mach_all)
ŷ_all = MLJ.predict(mach_all, X)
baseline_r2 = round(rsquared(ŷ_all, y), sigdigits=4)
println("R² with all features: $baseline_r2")
println("Features used: $(names(X))")
println()

# ## Method 1: Forward Selection
println("=== METHOD 1: FORWARD SELECTION ===")

function forward_selection(X, y, model)
    available_features = names(X)
    selected_features = String[]
    best_r2 = 0.0
    results = []
    
    while !isempty(available_features)
        best_feature = ""
        best_r2_iter = 0.0
        
        # Try adding each remaining feature
        for feature in available_features
            test_features = [selected_features; feature]
            X_test = select(X, Symbol.(test_features))
            
            mach_test = machine(model, X_test, y)
            fit!(mach_test)
            ŷ_test = MLJ.predict(mach_test, X_test)
            r2_test = rsquared(ŷ_test, y)
            
            if r2_test > best_r2_iter
                best_r2_iter = r2_test
                best_feature = feature
            end
        end
        
        # Check if adding this feature improves R²
        if best_r2_iter > best_r2
            push!(selected_features, best_feature)
            Base.filter!(x -> x != best_feature, available_features)
            best_r2 = best_r2_iter
            push!(results, (features=copy(selected_features), r2=best_r2))
            println("Added feature: $best_feature, R² = $(round(best_r2, sigdigits=4))")
        else
            break  # No improvement, stop
        end
    end
    
    return results
end

forward_results = forward_selection(X, y, model)
best_forward = forward_results[end]
println("Best forward selection model:")
println("Features: $(best_forward.features)")
println("R²: $(round(best_forward.r2, sigdigits=4))")
println()

# ## Method 2: Backward Elimination
println("=== METHOD 2: BACKWARD ELIMINATION ===")

function backward_elimination(X, y, model; significance_threshold=0.05)
    current_features = names(X)
    results = []
    
    while length(current_features) > 1
        # Fit model with current features
        X_current = select(X, Symbol.(current_features))
        mach_current = machine(model, X_current, y)
        fit!(mach_current)
        ŷ_current = MLJ.predict(mach_current, X_current)
        current_r2 = rsquared(ŷ_current, y)
        
        push!(results, (features=copy(current_features), r2=current_r2))
        
        # Try removing each feature and see impact on R²
        worst_feature = ""
        smallest_r2_drop = Inf
        
        for feature in current_features
            temp_features = Base.filter(x -> x != feature, current_features)
            if isempty(temp_features)
                continue
            end
            
            X_temp = select(X, Symbol.(temp_features))
            mach_temp = machine(model, X_temp, y)
            fit!(mach_temp)
            ŷ_temp = MLJ.predict(mach_temp, X_temp)
            r2_temp = rsquared(ŷ_temp, y)
            
            r2_drop = current_r2 - r2_temp
            if r2_drop < smallest_r2_drop
                smallest_r2_drop = r2_drop
                worst_feature = feature
            end
        end
        
        # Remove the least important feature (smallest R² drop)
        Base.filter!(x -> x != worst_feature, current_features)
        println("Removed feature: $worst_feature, R² drop: $(round(smallest_r2_drop, sigdigits=4))")
    end
    
    return results
end

backward_results = backward_elimination(X, y, model)
# Find the model with highest R² from backward elimination
best_backward = backward_results[argmax([r.r2 for r in backward_results])]
println("Best backward elimination model:")
println("Features: $(best_backward.features)")
println("R²: $(round(best_backward.r2, sigdigits=4))")
println()

# ## Method 3: Exhaustive Search (for small subsets)
println("=== METHOD 3: EXHAUSTIVE SEARCH (subsets of size 1-6) ===")

function exhaustive_search(X, y, model; max_features=6)
    all_features = names(X)
    best_models = []
    
    for k in 1:min(max_features, length(all_features))
        best_r2_k = 0.0
        best_features_k = String[]
        
        # Try all combinations of k features
        for feature_combo in combinations(all_features, k)
            X_combo = select(X, Symbol.(feature_combo))
            mach_combo = machine(model, X_combo, y)
            fit!(mach_combo)
            ŷ_combo = MLJ.predict(mach_combo, X_combo)
            r2_combo = rsquared(ŷ_combo, y)
            
            if r2_combo > best_r2_k
                best_r2_k = r2_combo
                best_features_k = collect(feature_combo)
            end
        end
        
        push!(best_models, (num_features=k, features=best_features_k, r2=best_r2_k))
        println("Best $k-feature model: R² = $(round(best_r2_k, sigdigits=4)), Features: $best_features_k")
    end
    
    return best_models
end

exhaustive_results = exhaustive_search(X, y, model, max_features=4)
best_exhaustive = exhaustive_results[argmax([r.r2 for r in exhaustive_results])]
println("Overall best from exhaustive search:")
println("Features: $(best_exhaustive.features)")
println("R²: $(round(best_exhaustive.r2, sigdigits=4))")
println()

# ## Method 4: LASSO Regularization (Automatic Feature Selection)
println("=== METHOD 4: LASSO REGULARIZATION ===")

# Load LASSO model  
LassoModel = @load LassoRegressor pkg = MLJLinearModels verbosity=0

# Try different regularization strengths
lambda_values = [0.01, 0.1, 1.0]  # Reduced from 7 values to 3
lasso_results = []

for lambda in lambda_values
    lasso = LassoModel(lambda=lambda)
    mach_lasso = machine(lasso, X, y)
    fit!(mach_lasso)
    ŷ_lasso = MLJ.predict(mach_lasso, X)
    r2_lasso = rsquared(ŷ_lasso, y)
    
    # Get non-zero coefficients (selected features)
    fp_lasso = fitted_params(mach_lasso)
    selected_features_lasso = [string(name) for (name, coef) in fp_lasso.coefs if abs(coef) > 1e-10]
    
    push!(lasso_results, (lambda=lambda, features=selected_features_lasso, r2=r2_lasso))
    println("λ = $lambda: R² = $(round(r2_lasso, sigdigits=4)), Features: $selected_features_lasso")
end

best_lasso = lasso_results[argmax([r.r2 for r in lasso_results])]
println("Best LASSO model:")
println("λ = $(best_lasso.lambda), Features: $(best_lasso.features)")
println("R²: $(round(best_lasso.r2, sigdigits=4))")
println()

# ## Method 5: Cross-Validation for Robust Evaluation
println("=== METHOD 5: CROSS-VALIDATION EVALUATION ===")

function evaluate_with_cv(X, y, model, features; cv_folds=5)
    X_subset = select(X, Symbol.(features))
    
    # Perform k-fold cross-validation
    cv = CV(nfolds=cv_folds, shuffle=true, rng=123)
    mach_cv = machine(model, X_subset, y)
    
    cv_results = evaluate!(mach_cv, resampling=cv, measure=rsquared)
    mean_r2 = cv_results.measurement[1]
    std_r2 = std(cv_results.per_fold[1])
    
    return mean_r2, std_r2
end

# Evaluate top candidates with cross-validation
candidates = [
    ("Forward Selection", best_forward.features),
    ("Backward Elimination", best_backward.features),
    ("Exhaustive Search", best_exhaustive.features),
    ("LASSO", best_lasso.features),
    ("All Features", names(X))
]

cv_results = []
for (method, features) in candidates
    if !isempty(features)
        mean_r2, std_r2 = evaluate_with_cv(X, y, model, features)
        push!(cv_results, (method=method, features=features, mean_r2=mean_r2, std_r2=std_r2))
        println("$method: CV R² = $(round(mean_r2, sigdigits=4)) ± $(round(std_r2, sigdigits=4))")
    end
end

# Find the best model based on cross-validation
best_cv = cv_results[argmax([r.mean_r2 for r in cv_results])]
println()
println("=== FINAL BEST MODEL ===")
println("Method: $(best_cv.method)")
println("Features: $(best_cv.features)")
println("Cross-validated R²: $(round(best_cv.mean_r2, sigdigits=4)) ± $(round(best_cv.std_r2, sigdigits=4))")

# ## Final Model Training and Visualization
best_features = best_cv.features
X_best = select(X, Symbol.(best_features))
mach_best = machine(model, X_best, y)
fit!(mach_best)
ŷ_best = MLJ.predict(mach_best, X_best)
final_r2 = rsquared(ŷ_best, y)

println("Training R² with best features: $(round(final_r2, sigdigits=4))")
println("Number of features: $(length(best_features))")
println()

# Show feature coefficients
fp_best = fitted_params(mach_best)
println("Feature coefficients:")
for (name, val) in fp_best.coefs
    println("$(rpad(name, 8)):  $(round(val, sigdigits=3))")
end
println("Intercept: $(round(fp_best.intercept, sigdigits=3))")

# Create comparison plot
actual_vs_predicted_plot = scatter(y, ŷ_best, 
    xlabel="Actual MedV", 
    ylabel="Predicted MedV",
    title="Best Model: Actual vs Predicted",
    legend=false,
    size=(800, 600),
    alpha=0.6)

# Add perfect prediction line
min_val, max_val = extrema([y; ŷ_best])
plot!(actual_vs_predicted_plot, [min_val, max_val], [min_val, max_val], 
    color=:red, linewidth=2, linestyle=:dash)

actual_vs_predicted_plot

# Summary comparison table
println("\n=== SUMMARY OF ALL METHODS ===")
println("Method                  | Features | R²")
println("------------------------|----------|--------")
println("Baseline (All)          | $(length(names(X)))        | $baseline_r2")
println("Forward Selection       | $(length(best_forward.features))        | $(round(best_forward.r2, sigdigits=4))")
println("Backward Elimination    | $(length(best_backward.features))        | $(round(best_backward.r2, sigdigits=4))")
println("Exhaustive Search       | $(length(best_exhaustive.features))        | $(round(best_exhaustive.r2, sigdigits=4))")
println("LASSO (λ=$(best_lasso.lambda))          | $(length(best_lasso.features))        | $(round(best_lasso.r2, sigdigits=4))")
println("Best (CV validated)     | $(length(best_cv.features))        | $(round(best_cv.mean_r2, sigdigits=4))")