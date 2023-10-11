
using Lux, LuxCUDA, Random, Zygote, MLUtils, Optimisers, ProgressBars, Printf, JLD2, StatsBase, NamedTupleTools, Plots

include("math.jl")
include("utils.jl")
include("layers.jl")
expr_size = 14
res_size = 5
features = 48
nheads = 3
warmup = 500
seq_len = expr_size + res_size
batch_size = 512
learning_rate = 6.0f-4

dev = gpu_device()


function create_mask(x)
    # x shape: (seq_len, batch_size)
    mask = (x .== 1)
    mask_expanded = reshape(mask, 1, size(mask, 1), 1, size(mask, 2))
    mask = repeat(mask_expanded, size(mask, 1), 1, 1, 1)
    return mask
end


function create_decoder()
    matrix_size = length(instances(MathToken))
    rng = Random.default_rng()
    model = Decoder(
        Dense(matrix_size => features),
        PositionalEncodingLayer(features, seq_len),
        Chain([
            DecoderBlock(
                MultiHeadAttention(
                    Dense(features => features),
                    Dense(features => features),
                    Dense(features => features),
                    Dropout(0.1),
                    Dense(features => features),
                    nheads
                ),
                LayerNorm((features, 1)),
                Chain(Dense(features => features * 4, gelu), Dense(features * 4 => features), Dropout(0.1)),
                Dropout(0.1),
                LayerNorm((features, 1))
            )
            for i in 1:3
        ]),
        Dense(features * seq_len => matrix_size),
    )
    ps, st = Lux.LuxCore.setup(rng, model)
    return model, ps, st
end

function create_decoder_simple()
    matrix_size = length(instances(MathToken))
    rng = Random.default_rng()
    model = Decoder(
        Dense(matrix_size => features),
        PositionalEncodingLayer(features, seq_len),
        Chain([
            NoOpLayer(),
        ]),
        Chain(
            Dense(features * seq_len => features * seq_len, relu),
            Dropout(0.2),
            Dense(features * seq_len => features * seq_len, relu),
            Dropout(0.2),
            Dense(features * seq_len => features * seq_len, relu),
            Dropout(0.2),
            Dense(features * seq_len => matrix_size),)
    )
    ps, st = Lux.LuxCore.setup(rng, model)
    return model, ps, st
end




"""
bias, scale の重みのWeightDecayを再帰的に0にする
"""
function set_weightdecay(t, learning_rate)
    chache = (;)
    for k in keys(t)
        if t[k] isa Optimisers.Leaf
            if k == :bias || k == :scale
                chache = merge_recursive(chache, (; k => Optimisers.Leaf(OptimiserChain(Adam(learning_rate, (0.9, 0.95), 1.0e-8), WeightDecay(0)), t[k].state,)),)
            end
        else
            chache = merge_recursive(chache, (; k => set_weightdecay(t[k], learning_rate)))
        end
    end
    chache
end

function crossentropy(y, ŷ)
    return mean(-sum(y .* logsoftmax(ŷ), dims=1))
end

create_optim(learning_rate, ps) = Optimisers.setup(Optimisers.AdamW(learning_rate, (0.9, 0.95), learning_rate * 0.1), ps)

mutable struct Model
    model
    ps
    st
end



function process_exprs(exprs)
    data = Vector{Tuple{Vector{Int},Int}}()

    for expr in exprs
        x = ones(Int, seq_len)
        if length(expr) > expr_size
            continue
        end
        expr_length = length(expr)
        x[1:expr_length] = Int.(expr)
        x[expr_size] = Int(Equal)
        res = calc_expr(expr)
        res_length = length(res)
        if res_length > res_size
            continue
        end
        for i in 0:res_length-1
            x[expr_size+1:expr_size+i] = Int.(res[1:i])
            push!(data, (copy(x), Int(res[i+1])))
        end
        x[expr_size+1:expr_size+res_length] = Int.(res[1:end])
        push!(data, (copy(x), Int(End)))
    end
    data
end


function evaluate(model, ps, st, exprs)
    cpu = cpu_device()
    datasize = length(exprs)

    st = LuxCore.testmode(st)
    ps, st = (ps, st) .|> dev
    success_count = 0
    for l in 1:datasize
        x = ones(Int, (seq_len, 1))
        expr = exprs[l]
        expr_length = length(expr)
        x[1:expr_length, 1] = expr .|> Int
        x[expr_size, 1] = Int(Equal)
        y = calc_expr(expr)
        y_length = length(y)
        res_vector = []
        for j in 1:y_length
            x_mask = create_mask(x)
            res, _ = model((onehotbacth(x), x_mask) .|> dev, ps, st) .|> cpu
            token = MathToken(argmax(vec(res)))
            if token == End
                break
            end
            push!(res_vector, token)
            x[expr_size+j, 1] = Int(y[j])

        end
        if !isempty(res_vector) && length(y) == length(res_vector) && all(y .== res_vector)
            success_count += 1
        end
    end
    success_count / datasize
end

function evaluate_bacth(model, ps, st, x, y)
    datasize = size(x, 2) ÷ res_size
    st = LuxCore.testmode(st)
    ps, st = (ps, st) .|> dev
    success_count = 0

    batch_size = 512
    res = batch_predict(model, ps, st, (onehotbacth(x), create_mask(x)), batch_size)
    _, predicted = findmax(res, dims=1)
    for i in 1:datasize
        for j in 1:res_size
            index = (i - 1) * res_size + j

            if y[1, index] != Int(Pad) && y[1, index] != predicted[1, index][1]
                break
            end
            if j == res_size
                success_count += 1
            end
        end
    end
    success_count / datasize
end

function training()
    rng = Random.MersenneTwister(1234)
    exprs = enumerate_simple_exprs(2)
    shuffle!(rng, exprs)
    train_exprs = exprs[end-length(exprs)÷4+1:end]
    train_data = process_exprs(train_exprs)
    train_x = hcat([b[1] for b in train_data]...)
    train_y = hcat([b[2] for b in train_data]...)
    test_exprs = exprs[1:end-length(exprs)÷4]
    test_data = process_exprs(test_exprs)
    test_x = hcat([b[1] for b in test_data]...)
    test_y = hcat([b[2] for b in test_data]...)

    matrix_size = length(instances(MathToken))
    model, ps, st = create_decoder()
    display(model)
    ps, st = (ps, st) .|> dev
    optim = create_optim(learning_rate, ps)
    optim = merge_recursive(optim, set_weightdecay(optim, learning_rate))
    model = Model(model, ps, st)
    iter = ProgressBar(1:3000)
    train_acc_list = Float64[]
    test_acc_list = Float64[]
    open("loss.txt", "w") do f
        println(f, "train_acc, test_acc")
    end
    for index in iter
        # ミニバッチで学習する
        num_batches = ceil(Int, length(train_data) / batch_size)
        trainingloss = 0.0
        for i in 1:num_batches
            start_idx = (i - 1) * batch_size + 1
            end_idx = min(i * batch_size, length(train_data))
            x = train_x[:, start_idx:end_idx]
            y = train_y[:, start_idx:end_idx]
            mask = create_mask(x)
            data = (onehotbacth(x), mask) .|> dev
            # Padの部分は無視する
            y = [j == 1 ? 0.0f0 : Float32(y[k] == j) for j in 1:matrix_size, k in axes(y, 2)] |> dev
            (loss, st), back = pullback(model.ps) do ps
                ŷ, _st = model.model(data, ps, model.st)
                crossentropy(y, ŷ), _st
            end
            trainingloss += loss
            gs = back((one(loss), nothing))[1]
            optim, ps = Optimisers.update(optim, ps, gs)
            model.st = st
            model.ps = ps
        end
        trainingloss /= num_batches
        test_acc = evaluate_bacth(model.model, model.ps, model.st, test_x, test_y)
        train_acc = evaluate_bacth(model.model, model.ps, model.st, train_x, train_y)

        set_description(iter, string(@sprintf("Loss: %9.4g", trainingloss)))
        open("loss.txt", "a") do f
            println(f, train_acc, ',', test_acc)
        end

        push!(train_acc_list, train_acc)
        push!(test_acc_list, test_acc)

        # プロットの更新
        plot(train_acc_list; label="Training Acc")
        plot!(test_acc_list; label="Test Acc")
        title!("Real-time Acc Plot")
        xlabel!("Epochs")
        ylabel!("Acc")
        display(current())
        try
            jldsave("model.jld2", ps=model.ps |> cpu_device(), st=model.st |> cpu_device())
        catch
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    training()
end