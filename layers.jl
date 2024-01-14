
struct PositionalEncodingLayer <: Lux.LuxCore.AbstractExplicitLayer
    features::Int
    seq_len::Int
end

Lux.initialparameters(rng::AbstractRNG, layer::PositionalEncodingLayer) = NamedTuple()

function Lux.initialstates(rng::AbstractRNG, layer::PositionalEncodingLayer)
    features = layer.features
    seq_len = layer.seq_len
    pos_enc = [
        i % 2 == 1 ? sin(j / 10000^(2 * i / features)) : cos(j / 10000^(2 * i / features))
        for i in 1:features, j in 1:seq_len, _ in 1:1
    ]
    (pos_enc=pos_enc,)
end
Lux.parameterlength(l::PositionalEncodingLayer) = 0
Lux.statelength(l::PositionalEncodingLayer) = l.features * l.seq_len

function (::PositionalEncodingLayer)(x, ps, st)
    return x .+ st.pos_enc, st
end

struct MultiHeadAttention <: Lux.LuxCore.AbstractExplicitContainerLayer{(:query, :key, :value, :dropout, :output)}
    query
    key
    value
    dropout
    output
    nheads
end

function (m::MultiHeadAttention)((x, mask), ps, st)
    # x = (features, seq_len, batch_size)
    q, _ = m.query(x, ps.query, st.query)
    k, _ = m.key(x, ps.key, st.key)
    v, _ = m.value(x, ps.value, st.value)
    st_dropout = nothing
    function fdrop(x)
        x, st_dropout = m.dropout(x, ps.dropout, st.dropout)
        x
    end
    values, _ = dot_product_attention(q, k, v; mask=mask, fdrop=fdrop, nheads=m.nheads)

    output, _ = m.output(values, ps.output, st.output)
    st = merge(st, (dropout=st_dropout,))
    return output, st
end

struct DecoderBlock <: Lux.LuxCore.AbstractExplicitContainerLayer{(:mha, :layer_norm1, :ffn, :dropout, :layer_norm2)}
    mha
    layer_norm1
    ffn
    dropout
    layer_norm2
end

function (m::DecoderBlock)((x, mask), ps, st)
    norm_out, st_layer_norm1 = m.layer_norm1(x, ps.layer_norm1, st.layer_norm1)
    attn_output, st_mha = m.mha((norm_out, mask), ps.mha, st.mha)
    x = attn_output + x

    norm_out, st_layer_norm2 = m.layer_norm2(x, ps.layer_norm2, st.layer_norm2)
    ffn_output, _ = m.ffn(norm_out, ps.ffn, st.ffn)
    dropout_out, st_dropout = m.dropout(ffn_output, ps.dropout, st.dropout)
    x = dropout_out + x

    st = merge(st, (mha=st_mha, layer_norm1=st_layer_norm1, layer_norm2=st_layer_norm2, dropout=st_dropout,))
    return (x, mask), st
end


struct Decoder <: Lux.LuxCore.AbstractExplicitContainerLayer{(:embedding, :positional_encoding, :blocks, :output)}
    embedding
    positional_encoding
    blocks
    output
end

function (m::Decoder)((x, mask), ps, st)
    # x = (token_size, seq_len, batch_size)
    x, _ = m.embedding(x, ps.embedding, st.embedding)
    x, _ = m.positional_encoding(x, ps.positional_encoding, st.positional_encoding)
    # x = (features, seq_len, batch_size)

    (x, _), st_blocks = m.blocks((x, mask), ps.blocks, st.blocks)

    output, st_out = m.output(x, ps.output, st.output)
    # output = (token_size, seq_len, batch_size)
    st = merge(st, (blocks=st_blocks, output=st_out))
    return output, st
end

function TransformerBlock(features,  nheads)
   Chain(
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
   )
end

struct MLPDecoder <: Lux.LuxCore.AbstractExplicitContainerLayer{(:embedding, :positional_encoding, :blocks, :output)}
    embedding
    positional_encoding
    blocks
    output
end

function (m::MLPDecoder)((x, _), ps, st)
    # x = (token_size, seq_len, batch_size)
    x, _ = m.embedding(x, ps.embedding, st.embedding)
    x, _ = m.positional_encoding(x, ps.positional_encoding, st.positional_encoding)
    # x = (features, seq_len, batch_size)

    x, st_blocks = m.blocks(x, ps.blocks, st.blocks)

    output, st_out = m.output(x, ps.output, st.output)
    # output = (token_size, seq_len, batch_size)
    st = merge(st, (blocks=st_blocks, output=st_out))
    return output, st
end



function MLPMixierBlock(features, seq_len)
    token_mixier = Chain(
        LayerNorm((features, seq_len)),
        WrappedFunction(x -> permutedims(x, (2, 1, 3))),
        Dense(seq_len => 4seq_len),
        gelu,
        Dense(4seq_len => seq_len),
        WrappedFunction(x -> permutedims(x, (2, 1, 3))),)
    channel_mixier = Chain(
        LayerNorm((features, seq_len)),
        Dense(features => 4features),
        gelu,
        Dense(4features => features),
    )
    Chain(SkipConnection(token_mixier, +), SkipConnection(channel_mixier, +))
end

function gMLPMixierBlock(features, seq_len)
    block = Chain(
        LayerNorm((features, seq_len)),
        BranchLayer(
            Dense(features => features * 3, gelu),
            Dense(features => features * 3, gelu),
        ),
        Parallel(
            .*,
            Chain(
                LayerNorm((features * 3, seq_len)),
                WrappedFunction(x -> permutedims(x, (2, 1, 3))),
                Dense(seq_len => seq_len;init_weight=zeros32, init_bias=ones32),
                WrappedFunction(x -> permutedims(x, (2, 1, 3))),
            ),
            NoOpLayer(),
        ),
        Dense(features * 3 => features),
    )

    SkipConnection(block, +)
end

function aMLPMixierBlock(features, seq_len)
    tiny_attention = Chain(
        BranchLayer(
            BranchLayer(
                Dense(features => features),
                Dense(features => features),
            ),
            NoOpLayer(),
        ),
        Parallel(
            # (a,b)-> ein"mnb,dmb -> dnb"(a,b),
            (a,b)->batched_mul(b,a), 
            Chain(
                Parallel(
                #    (a,b)-> ein"dnb,dmb -> mnb"(a,b),
                (a, b) -> batched_mul(permutedims(b, (2, 1, 3)), a),
                    NoOpLayer(),
                    NoOpLayer(),
                ),
                WrappedFunction(x -> softmax(x ./ √features)),
            ),
            NoOpLayer(),
        ),
        Dense(features => 2features),
    )
    block = Chain(
        LayerNorm((features, seq_len)),
        BranchLayer(
            Dense(features => features * 2, gelu),
            BranchLayer(
                Dense(features => features * 2, gelu),
                Dense(features => features, gelu),
            ),
        ),
        Parallel(
            .*,
            NoOpLayer(),
            Parallel(.+,
                Chain(
                    LayerNorm((features * 2, seq_len)),
                    WrappedFunction(x -> permutedims(x, (2, 1, 3))),
                    Dense(seq_len => seq_len; init_bias=ones32),
                    WrappedFunction(x -> permutedims(x, (2, 1, 3))),
                ),
                tiny_attention,
            ),
        ),
        Dense(features * 2 => features),
    )

    SkipConnection(block, +)
end