import PyPlot
import Distributions

struct impulse_data
  x::Array{Float64, 2}
  y::Array{Float64, 1}

  function impulse_data(data_len::Int64)
    x = zeros(Float64, data_len, 2)
    y = zeros(Float64, data_len)

    period::Int64 = 11
    i::Int64 = 1
    pulse_len::Int64 = 0

    while i <= data_len
      a = rand(0:1)
      x[i, 1] = a
      if a == 0
        b = rand()
        x[i, 2] = b
        pulse_len = round(b * 10)
        if i + pulse_len <= data_len
          y[i:(i + pulse_len)] = 1
        else
          y[i:end] = 1
        end
        i += period
      end
      i += 1
    end

    new(x, y)
  end

end

function activation(x::Array{Float64, 1})::Array{Float64, 1}
  t = ones(length(x))
  return t./(t + exp(-x))
end

function forvard_pass(x::Array{Float64, 1}, W_0::Array{Float64, 2},
                      W_1::Array{Float64, 2}, W_h::Array{Float64, 2},
                      h_prev::Array{Float64, 1})
  v_0 = W_0 * x
  v_h = W_h * h_prev
  h = activation(v_0 + v_h)
  v1= W_1 * h
  y = activation(v1)

  return y, h
end

function activation_derivative(output::Array{Float64, 1})::Array{Float64, 1}
  return output .* (ones(length(output)) - output)
end

function main()
  train_data_len::Int64 = 200000
  train_data = impulse_data(train_data_len)

  valid_data_len = 200
  valid_data = impulse_data(valid_data_len)

  nu       = 0.001
  input_dim  = 2
  hidden_dim = 180
  output_dim = 1

  W_0 = Distributions.rand(Distributions.Uniform(-0.9, 0.9), hidden_dim, input_dim)
  W_1 = Distributions.rand(Distributions.Uniform(-0.9, 0.9), output_dim, hidden_dim)
  W_h = Distributions.rand(Distributions.Uniform(-0.9, 0.9), hidden_dim, hidden_dim)

  d_0_l = zeros(hidden_dim, input_dim)
  d_1_l = zeros(output_dim, hidden_dim)
  d_h_l = zeros(hidden_dim, hidden_dim)
  backprop_depth = 20
  inputs_unrolled = Array{Array{Float64, 1}, 1}(backprop_depth)
  h_unrolled      = Array{Array{Float64, 1}, 1}(backprop_depth)
  d_0_unrolled    = Array{Array{Float64, 1}, 1}(backprop_depth)
  d_1_unrolled    = Array{Array{Float64, 1}, 1}(backprop_depth)
  for i = 1:backprop_depth
    inputs_unrolled[i] = zeros(Float64, input_dim)
    h_unrolled[i] = zeros(Float64, hidden_dim)
    d_0_unrolled[i] = zeros(Float64, hidden_dim)
    d_1_unrolled[i] = zeros(Float64, output_dim)
  end
  for j = 1:train_data_len
    sample = train_data.x[j, :]
    t = train_data.y[j]
    h_prev = h_unrolled[end]
    y_1, h = forvard_pass(sample, W_0, W_1, W_h, h_prev)

    # backward pass
    e = t - y_1
    d_1 = e .* activation_derivative(y_1)
    d_0 = (W_1' * d_1) .* activation_derivative(h)

    d_0_unrolled = d_0_unrolled[2:end]
    push!(d_0_unrolled, d_0)
    d_1_unrolled = d_1_unrolled[2:end]
    push!(d_1_unrolled, d_1)
    h_unrolled = h_unrolled[2:end]
    push!(h_unrolled, h)
    inputs_unrolled = inputs_unrolled[2:end]
    push!(inputs_unrolled, sample)

    for l = 1:backprop_depth
      d_1_l += d_1_unrolled[l] * h_unrolled[l]'
      d_h_l += d_0_unrolled[l] * h_unrolled[l]'
      d_0_l += d_0_unrolled[l] * inputs_unrolled[l]'
    end

    W_0 += nu .* d_0_l
    W_1 += nu .* d_1_l
    W_h += nu .* d_h_l
    d_0_l *= 0
    d_1_l *= 0
    d_h_l *= 0
  end

  actual_out = zeros(Float64, valid_data_len)
  h_prev = zeros(Float64, hidden_dim)
  for j = 1:valid_data_len
    sample = valid_data.x[j, :]
    #t = valid_data.y[j]
    y_1, h = forvard_pass(sample, W_0, W_1, W_h, h_prev)
    #e = t - y_1 # TODO plot e, and validation data
    h_prev = copy(h)
    actual_out[j] = y_1[1]
  end

#  PyPlot.plot(valid_data.y)
#  PyPlot.plot(actual_out)
#  PyPlot.show()
end

main()
