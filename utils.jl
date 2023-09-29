function onehotbacth(x)
    # x shape: (seq_len, batch_size)
    matrix_size = length(instances(MathToken))
    res = zeros(matrix_size, size(x, 1), size(x, 2))

    for j in axes(x, 2)
        for i in axes(x, 1)
            res[x[i, j], i, j] = 1
        end
    end
    return res
end

function loadmodel()
    model, _, _ = create_decoder()
    data = load("model.jld2")
    ps = data["ps"]
    st = data["st"]
    model, ps, st
end

"""
バッチサイズを指定して予測を行う
"""
function batch_predict(model, ps, st, data, batch_size)
    cpu = cpu_device()
    x, mask = data

    n = size(x, 3) 
    num_batches = ceil(Int, n / batch_size)
    responses = []

    for i in 1:num_batches
        start_idx = (i - 1) * batch_size + 1
        end_idx = min(i * batch_size, n)

        x_batch = x[:, :, start_idx:end_idx] |> dev
        mask_batch = mask[:, :, :, start_idx:end_idx] |> dev

        response_batch, _ = model((x_batch, mask_batch), ps, st) |> cpu
        push!(responses, response_batch)
    end

    return hcat(responses...)
end

"""
単一の式をAIに計算させる
"""
function calc_ai(expr)
    model, ps, st = loadmodel()
    st = LuxCore.testmode(st)
    ps, st = (ps, st) .|> dev
    x = ones(Int, (seq_len, 1))
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
    return res_vector
end
