export FluxPartModel_Q10

# Define the model structure (inheriting from EasyHybridModels)
struct FluxPartModel_Q10 <: EasyHybridModels
    RUE_chain::Flux.Chain
    RUE_predictors::AbstractArray{Symbol}
    Rb_chain::Flux.Chain
    Rb_predictors::AbstractArray{Symbol}
    predictors::AbstractArray{Symbol}
    forcing::AbstractArray{Symbol}
    Q10
end


"""
FluxPartModel_Q10(RUE_predictors, Rb_predictors; Q10=[1.5f0])
"""
# Combined constructor and inference function (no overloading)
function FluxPartModel_Q10_infer(dk, RUE_predictors, Rb_predictors; forcing=[:SW_IN, :TA], Q10=[1.5f0], neurons=15)
    # Step 1: Construct the neural networks
    RUE_ch = Dense_RUE_Rb(length(RUE_predictors); neurons)  # RUE neural network
    Rb_ch = Dense_RUE_Rb(length(Rb_predictors); neurons)    # Rb neural network
    
    # Create the model object (to be used in inference)
    model = FluxPartModel_Q10(
        RUE_ch,              # RUE_chain neural network
        RUE_predictors,      # Predictors for RUE
        Rb_ch,               # Rb_chain neural network
        Rb_predictors,       # Predictors for Rb
        union(RUE_predictors, Rb_predictors), # Union of both predictors
        forcing,             # Forcing variables (like SW_IN, TA)
        Q10                  # Q10 parameter for temperature dependence
    )
    
    # Step 2: Perform inference (calculate RUE, Rb, GPP, Reco, NEE)
    
    # Select predictors for the neural networks from the input data (dk)
    RUE_input4Chain = select_predictors(dk, RUE_predictors)
    Rb_input4Chain = select_predictors(dk, Rb_predictors)
    
    # Forward pass through the neural networks (RUE_chain and Rb_chain)
    Rb = 100.0f0 * model.Rb_chain(Rb_input4Chain)
    RUE = 1.0f0 * model.RUE_chain(RUE_input4Chain)
    
    # Extract forcing variables (SW_IN and TA) from the data
    sw_in = select_variable(dk, forcing[1])
    ta = select_variable(dk, forcing[2])

    # Calculate GPP and Reco based on the model outputs
    GPP = sw_in .* RUE ./ 12.011f0
    Reco = Rb .* Q10[1] .^ (0.1f0(ta .- 15.0f0))
    
    # Return a named tuple with results (GPP, Reco, NEE, etc.)
    return (; RUE, Rb, GPP=GPP, RECO=Reco, NEE=Reco - GPP)
end

# This enables Flux to access the parameters in the model for training
#Flux.@functor is a macro provided by the Flux library. It automatically makes the fields of your struct 
# (in this case, FluxPartModel_Q10) trainable by Flux. This means that when you use Flux’s optimization 
# or backpropagation tools, the parameters (such as the weights in RUE_chain and Rb_chain) can be updated.
Flux.@functor FluxPartModel_Q10