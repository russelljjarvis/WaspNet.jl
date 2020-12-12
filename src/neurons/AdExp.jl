"""
struct ADEXP{T<:Number}<:AbstractNeuronn

Contains the vector of paramters [a, b, c, d, I, θ] necessary to simulate an Izhikevich neuron as well as the current state of the neuron.

The @with_kw macro is used to produce a constructor which accepts keyword arguments for all values. This neuron struct is immutable, therefor we store the state of the neuron in an `Array` such that its values can change while the parameters remain static. This represents a minimal example for an `AbstractNeuron` implementation to build it into a `Layer`.


[Adaptive_exponential_integrate and fire neuron](http://www.scholarpedia.org/article/Adaptive_exponential_integrate-and-fire_model)
Dr. Wulfram Gerstner
Romain Brette, Ecole Normale Supérieure, Paris, France

# Fields
- `a::T`-`d::T`: Neuron parameters as described at https://www.izhikevich.org/publications/spikes.htm
- `I::T`: Background current (mA)
- `θ::T`: Threshold potential (mV)
- `v0::T`: Reset voltage (mV)
- `u0::T`: Reset recovery variable value
- `state::T`: Vector holding the current (v,u) state of the neuron
- `output::T`: Vector holding the current output of the neuron
"""

@with_kw struct ADEXP{T<:Number}<:AbstractNeuron
    a::T = 4.0
    b::T = 0.0805
    cm::T = 0.281
    v_rest::T = -70.6
    v::T = -70.6     # Membrane potential (mV)
    tau_m::T = 9.3667
    tau_w::T = 144.0
    v_thresh::T = -50.4
    delta_T::T = 2.0
    v_spike::T = -40.0
    v_reset::T = -70.6
    spike_delta::T = 30
    I::T = 25.

end



"""
update(neuron::Izh, input_update, dt, t)
Evolves the given `Neuron` subject to an input of `input_update` a time duration `dt` starting from time `t` according to the equations defined in the Izhikevich paper https://www.izhikevich.org/publications/spikes.htm
We use an Euler update for solving the set of differential equations for its computational efficiency and simplicity of implementation.
"""
function update(neuron::ADEXP, synaptic_input_update, dt, t)
    dt *= 1000. # convert seconds to milliseconds for the Adexp model
    fire = 0.
    @unpack a,b,cm,v_rest,v,tau_m,tau_w,v_thresh,delta_T,v_spike,v_reset,spike_delta,I = neuron
    v = v + synaptic_input_update
    if spike_raster[cnt] == 1 || fire
      v = v_reset
      w += b
    end
    dv  = (((v_rest-v) +
            delta_T*exp((v - v_thresh)/delta_T))/tau_m +
            (I - w)/cm) *dt
    v += dv
    w += dt * (a*(v - v_rest) - w)/tau_w * dt
    fire = v > v_thresh
    if v>v_thresh
        fire = 1
        v = spike_delta
        spike_raster[cnt] = fire
    else
        spike_raster[cnt] = 0
    end
    neuron.v = v
    return (fire, ADEXP(neuron.a, neuron.b, neuron.cm,
    neuron.v_rest, neuron.tau_m, neuron.tau_w, neuron.v_thresh,
    neuron.delta_T, neuron.v_spike, neuron.v_reset,neuron.spike_delta, fire))
end

"""
reset(neuron::Izh)
Resets the state of the Izhikevich neuron to its initial values given by `v0`, `u0`
"""
function reset(neuron::ADEXP)
    return ADEXP(
        neuron.a, neuron.b, neuron.cm,
        neuron.v_rest, neuron.tau_m, neuron.tau_w, neuron.v_thresh,
        neuron.delta_T, neuron.v_spike, neuron.v_reset,neuron.spike_delta, 0.
        )
end

function get_neuron_states(neuron::ADEXP)
    return (neuron.v, neuron.w)
end
