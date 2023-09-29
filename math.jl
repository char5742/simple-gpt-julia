@enum MathToken begin
    Pad = 1
    # 演算子
    Plus
    Times
    Equal
    # 数字
    Zero
    One
    Two
    Three
    Four
    Five
    Six
    Seven
    Eight
    Nine
    LeftParen
    RightParen
    End
end

const numbers = [Zero, One, Two, Three, Four, Five, Six, Seven, Eight, Nine]
const operators = [Plus, Times]


"数値トークンを数値に変換する関数"
function token_to_num(t::MathToken)::Int
    if t in numbers
        # オペレーター分のオフセットを引く
        return Int(t) - 5
    else
        throw(ArgumentError("t is not a number"))
    end
end

"数値を数値トークンに変換する関数"
function num_to_tokens(n::Number)::Vector{MathToken}
    # オペレーター分のオフセットを足す
    return [MathToken(v + 5) for v in reverse(digits(n))]
end

"トークンを文字列に変換する関数"
function token_to_string(t::MathToken)::String
    # オペレーター分のオフセットを足す
    if t in numbers
        return string(token_to_num(t))
    elseif t == Plus
        return "+"
    elseif t == Times
        return "*"
    elseif t == Equal
        return "="
    elseif t == LeftParen
        return "("
    elseif t == RightParen
        return ")"
    else
        throw(ArgumentError("t is not a number"))
    end
end

"数式の計算結果をトークンベクトルで返す関数"
function calc_expr(expr::Vector{MathToken})::Vector{MathToken}
    res = expr .|> token_to_string |> join |> Meta.parse |> eval
    num_to_tokens(res)
end

"""
全ての複雑な式を列挙する

下記のような式を全て列挙する  
(8+6)*7 = [LeftParen ,Eight, Plus, Six, RightParen, Times, Seven]

"""
function enumerate_complex_exprs(depth::Int)
    if depth == 0
        return [[n] for n in numbers]
    end

    exprs = Vector{Vector{MathToken}}()
    for expr1 in enumerate_complex_exprs(depth - 1)
        for expr2 in enumerate_complex_exprs(depth - 1)
            for op in operators
                push!(exprs, [expr1..., op, expr2...])
                push!(exprs, [LeftParen, expr1..., op, expr2..., RightParen])
            end
        end
    end

    return exprs
end

"""
全ての単純な式を列挙する

下記のような式を全て列挙する  
8+6 = [Eight, Plus, Six]
"""
function enumerate_simple_exprs(k::Int)
    res = Vector{Vector{MathToken}}()
    max_num = 10^k - 1
    for i in 0:max_num
        for j in 0:max_num
            expr = [num_to_tokens(i)..., Plus, num_to_tokens(j)...]
            push!(res, expr)
        end
    end
    res
end
