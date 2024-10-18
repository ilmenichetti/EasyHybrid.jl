export FluxPartModel_Q10


# defining the structure of the function, declaring the objects used and their types
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
# constructor function. This is the base of the function, even if it does not contain methods (which are called later, overloaded onto this one)
function FluxPartModel_Q10(RUE_predictors, Rb_predictors; forcing=[:SW_IN, :TA], Q10=[1.5f0], neurons=15)
    RUE_ch = Dense_RUE_Rb(length(RUE_predictors); neurons)
    Rb_ch = Dense_RUE_Rb(length(Rb_predictors); neurons)
    FluxPartModel_Q10(
        RUE_ch,
        RUE_predictors,
        Rb_ch,
        Rb_predictors,
        union(RUE_predictors, Rb_predictors),
        forcing,
        Q10
    )
end

#Model call section
#defining the methods, the actual functions. This is done by overloading these methods onto FluxPartModel_Q10, 
# and the new function becomes m.
#m is called onto the object kd (the data), but only when m is called with ":infer" as argument. 
#:: is the type assertion operator in Julia. It means that the argument on the left-hand side must have the type 
# specified on the right-hand side. This in Julia is called method dispatch, and it works as some kind of conditional
# code execution.
function (m::FluxPartModel_Q10)(dk, ::Val{:infer})
    RUE_input4Chain = select_predictors(dk, m.RUE_predictors)
    Rb_input4Chain = select_predictors(dk, m.Rb_predictors)
    Rb = 100.0f0 * m.Rb_chain(Rb_input4Chain)
    RUE = 1.0f0 * m.RUE_chain(RUE_input4Chain)
    #SW_IN = Matrix(x([:SW_IN]))
    #TA = Matrix(x([:TA]))
    #µmol/m2/s1 =  J/s/m2 * g/MJ / g/mol
    sw_in = select_variable(dk, m.forcing[1])
    ta = select_variable(dk, m.forcing[2])

    GPP = sw_in .* RUE ./ 12.011f0
    Reco = Rb .* m.Q10[1] .^ (0.1f0(ta .- 15.0f0))
    return (; RUE, Rb, GPP=GPP, RECO=Reco, NEE=Reco - GPP)
end

# overloaded call: calls the components above, computes and returns just the NEE
function (m::FluxPartModel_Q10)(dk)
    res = m(dk, :infer) #by using the :infer flag (which in Julia is called a symbol) this part calls the overloaded functions as well, not just the core
    return res.NEE
end

"""
(m::`FluxPartModel_Q10`)(dk, infer::Symbol)
"""
#this function is a helper function that takes a symbol as input and convert it to value type, it allows for
#flexibility when calling the function. Without this step the code in the model execution overloaded part would not run.
#If :infer is a symbol, the above overloaded model function would not run and then calling m(dk, :infer) is
#caught by this function.
#while we could use directly a symbol instead of a value for method dispatch, this would make for a much less efficient code.
# symbols are not part of the types system in Julia. This would mean the dispatch would not be evaluated in compile-time but
#only in runtime, and the Julia JIT compiler would not be able to evaluate in beforehand how the program should run, degrading performances.
function (m::FluxPartModel_Q10)(dk, infer::Symbol)
    return m(dk, Val(infer))
end

# Call @functor to allow for training the custom model
Flux.@functor FluxPartModel_Q10