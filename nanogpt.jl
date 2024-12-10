# Taken from https://github.com/FluxML/model-zoo/pull/410
using MLUtils, Lux, Random, Optimisers, Printf, Statistics, NNlib, DataDeps, StatsBase,
      OneHotArrays, JLD2
using Zygote, LuxCUDA, CUDA, LinearAlgebra, Plots
using Comonicon: @main

if !haskey(DataDeps.registry, "nanogpt")
    register(DataDep(
        "nanogpt",
        "Shakespeare Input Text for training NanoGPT",
        "https://cs.stanford.edu/people/karpathy/char-rnn/shakespeare_input.txt",
        "59a0ad62833b2e15ec811c548618876359e902717431236e52699a0e2bc253ca"
    ))
end

function gpt_block(; n_embed, n_hidden, qk_dim, v_dim, n_heads, dropout_rate)
    @assert qk_dim % n_heads == 0
    @assert v_dim % n_heads == 0
    return @compact(;
        name="GPTBlock(; n_embed=$n_embed, n_hidden=$n_hidden, qk_dim=$qk_dim, v_dim=$v_dim, n_heads=$n_heads, dropout_rate=$dropout_rate)",
        ln=LayerNorm((n_embed, 1)),
        qlayer=Dense(n_embed => qk_dim; use_bias=false, init_weight=glorot_normal),
        klayer=Dense(n_embed => qk_dim; use_bias=false, init_weight=glorot_normal),
        vlayer=Dense(n_embed => v_dim; use_bias=false, init_weight=glorot_normal),
        attn_drop=Dropout(dropout_rate),
        proj=Dense(v_dim => n_embed; use_bias=false, init_weight=glorot_normal),
        mlp=Chain(
            LayerNorm((n_embed, 1)),
            Dense(n_embed => n_hidden, gelu; init_weight=glorot_normal),
            Dense(n_hidden => n_embed; init_weight=glorot_normal),
            Dropout(dropout_rate)
        )) do x
        qkv_in = ln(x)
        q = qlayer(qkv_in)
        k = klayer(qkv_in)
        v = vlayer(qkv_in)
        mha, _ = NNlib.dot_product_attention(
            q, k, v, nothing; mask=NNlib.make_causal_mask(x)
        )
        x = x .+ proj(mha)
        x = x .+ mlp(x)
        @return x
    end
end

function GPT(;
        n_vocab, n_embed, sequence_length, n_hidden,
        n_layers, dropout_rate, n_heads, qk_dim, v_dim
)
    return @compact(;
        token_embedding=Embedding(n_vocab => n_embed),
        position_embedding=Embedding(sequence_length => n_embed),
        drop=Dropout(dropout_rate),
        blocks=Chain(ntuple(n_layers) do i
            return gpt_block(; n_embed, n_hidden, qk_dim, v_dim, n_heads, dropout_rate)
        end...),
        ln=LayerNorm((n_embed, 1)),
        output_layer=Dense(n_embed => n_vocab, init_weight=glorot_normal)) do tokens
        x = drop(token_embedding(tokens) .+ position_embedding(1:size(tokens, 1)))
        x = blocks(x)
        @return output_layer(ln(x))
    end
end

# Use the model to generate some text.
function generate_text(
        model, ps, st, seed; alphabet, output_length, sequence_length
)
    dev = get_device((ps, st))
    # @assert !(dev isa ReactantDevice) "Currently we don't support running inference of \
    #                                    dynamically sized tensors."

    seed = copy(seed)
    seed_len = maximum(length, seed)
    extra_letters = zeros(Int, length(seed))
    for (i, s) in enumerate(seed)
        if seed_len != length(s)
            extra_letters[i] = seed_len - length(s)
            seed[i] = "_"^extra_letters[i] * s
        end
    end
    original_output_length = output_length
    output_length += maximum(extra_letters)

    st = Lux.testmode(st)

    x = zeros(Int, output_length, length(seed))
    for (i, s) in enumerate(seed), j in 1:seed_len
        x[j, i] = findfirst(==(s[j]), alphabet)
    end
    for i in (seed_len + 1):output_length
        tail = x[max(1, i - sequence_length + 1):(i - 1), :] |> dev
        y = model(tail, ps, st)[1] |> cpu_device()
        p = softmax(y[:, end, 1])
        x[i, :] .= sample(1:length(alphabet), Weights(p))
    end

    res = [String(map(Base.Fix1(getindex, alphabet), x[:, i])) for i in axes(x, 2)]
    for i in eachindex(res)
        res[i] = res[i][(extra_letters[i] + 1):end][1:original_output_length]
    end

    return res
end

# Load data from input file, and partition into training and testing subsets.
function get_nanogpt_data(; sequence_length, test_split)
    data_file = joinpath("./shakespeare_input.txt")
    text = String(read(data_file))

    # For aesthetic reasons, replace newlines with strings.  This is not necessary, but makes
    # strings print nicer.
    text = replace(text, r"\r?\n" => " ")

    ## an array of all unique characters
    alphabet = [unique(text)..., '_']
    stop = alphabet[end]

    B = (length(text) - 1) ÷ sequence_length
    # We must collect() before indexing, because String indexing does strange things with multi-byte
    # characters and we could end up with the wrong length.
    Xs = reshape(collect(text)[1:(B * sequence_length)], sequence_length, B)
    Ys = reshape(collect(text)[2:(B * sequence_length + 1)], sequence_length, B)

    # Input string starts with stop character '_', representing zero context.
    Xs[1, :] .= stop

    # Xs (input) should consist of indices into `alphabet` because this is what Embedding expects.
    # Ys (output) should be one-hot because this is what logitcrossentropy expects.
    Xs = map(c -> Int32(findfirst(==(c), alphabet)), Xs)
    Ys = onehotbatch(Ys, alphabet)

    trainX, testX = MLUtils.splitobs(Xs; at=1 - test_split)
    trainY, testY = MLUtils.splitobs(Ys; at=1 - test_split)

    return alphabet, Array(trainX), Array(trainY), Array(testX), Array(testY)
end

eigs_gpt = (layer_1 = (layer_2 = [], layer_3 = []), layer_2 = (layer_2 = [], layer_3 = []), layer_3 = (layer_2 = [], layer_3 = []), 
            layer_4 = (layer_2 = [], layer_3 = []), layer_5 = (layer_2 = [], layer_3 = []), layer_6 = (layer_2 = [], layer_3 = []))

function eigengpt(ps)
    w1 = ps.blocks.layer_1.mlp.layer_2.weight
    m = size(w1, 1)
    println(size(w1))
    push!(eigs_gpt.layer_1.layer_2, svdvals((w1'*w1)./m))
    w2 = ps.blocks.layer_1.mlp.layer_3.weight
    m = size(w2, 2)
    println(size(w2))
    push!(eigs_gpt.layer_1.layer_3, svdvals((w2*w2')./m))
    
    w3 = ps.blocks.layer_2.mlp.layer_2.weight
    m = size(w3, 1)
    push!(eigs_gpt.layer_2.layer_2, svdvals((w3'*w3)/m))
    w4 = ps.blocks.layer_2.mlp.layer_3.weight
    m = size(w4, 2)
    push!(eigs_gpt.layer_2.layer_3, svdvals((w4*w4')/m))
    
    w1 = ps.blocks.layer_3.mlp.layer_2.weight
    m = size(w1, 1)
    push!(eigs_gpt.layer_3.layer_2, svdvals((w1'*w1)/m))
    w1 = ps.blocks.layer_3.mlp.layer_3.weight
    m = size(w1, 2)
    push!(eigs_gpt.layer_3.layer_3, svdvals((w1*w1')/m))
    
    w1 = ps.blocks.layer_4.mlp.layer_2.weight
    m = size(w1, 1)
    push!(eigs_gpt.layer_4.layer_2, svdvals((w1'*w1)/m))
    w1 = ps.blocks.layer_4.mlp.layer_3.weight
    m = size(w1, 2)
    push!(eigs_gpt.layer_4.layer_3, svdvals((w1*w1')/m))

    w1 = ps.blocks.layer_5.mlp.layer_2.weight
    m = size(w1, 1)
    push!(eigs_gpt.layer_5.layer_2, svdvals((w1'*w1)/m))
    w1 = ps.blocks.layer_5.mlp.layer_3.weight
    m = size(w1, 2)
    push!(eigs_gpt.layer_5.layer_3, svdvals((w1*w1')/m))

    w1 = ps.blocks.layer_6.mlp.layer_2.weight
    m = size(w1, 1)
    push!(eigs_gpt.layer_6.layer_2, svdvals((w1'*w1)/m))
    w1 = ps.blocks.layer_6.mlp.layer_3.weight
    m = size(w1, 2)
    push!(eigs_gpt.layer_6.layer_3, svdvals((w1*w1')/m))
end

# @main function main(;
n_embed::Int=128 
n_hidden::Int=512 
n_heads::Int=4 
qk_dim::Int=16
v_dim::Int=16
n_layers::Int=6 
sequence_length::Int=64 
batchsize::Int=64
dropout_rate::Float32=0.0f0 
test_split::Float64=0.1 
lr::Float64=5e-4
epochs::Int=50
# Only inference option
inference::Bool=false 
model_path::String=""
seed::Union{String, Vector{String}}=["_", "The", "Julia", "Lux.jl"]
output_length::Int=1024
# )
rng = Random.default_rng()
Random.seed!(rng, 1234)
dev = gpu_device()
cdev = cpu_device()
if inference
    @printf "[Info] Inference mode enabled.\n"
    @assert !isempty(model_path) "Please provide a path to a model checkpoint."
    @printf "[Info] Loading model from %s.\n" model_path
    model_config = JLD2.load(model_path, "model_config")
    model = GPT(; model_config...)
    ps = JLD2.load(model_path, "parameters")
    st = JLD2.load(model_path, "states")
    alphabet = JLD2.load(model_path, "alphabet")
    sequence_length = model_config.sequence_length
    texts = generate_text(
        model, ps, st, seed; alphabet, output_length, sequence_length
    )
    for (i, (text, s)) in enumerate(zip(texts, seed))
        @printf "[Info] Seed [%d]: %s\n" i s
        @printf "[Generated Text] %s\n\n" text
    end
    return
end

alphabet, trainX, trainY, testX, testY = get_nanogpt_data(; sequence_length, test_split)
@printf "[Info] Alphabet size: %d\n" length(alphabet)
@printf "[Info] Training size: %d sequences.\n" size(trainX, 2)
@printf "[Info] Testing  size: %d sequences.\n\n" size(testX, 2)
train_loader = DataLoader(
    (trainX, trainY); batchsize, shuffle=true, parallel=true
) .|> dev
model_config = (;
    n_vocab=length(alphabet), n_embed, sequence_length, n_hidden,
    n_layers, dropout_rate, n_heads, qk_dim, v_dim
)
model = GPT(; model_config...)
ps, st = Lux.setup(rng, model) |> dev
@printf "[Info] Number of parameters: %d\n" Lux.parameterlength(ps)
@printf "[Info] Number of states: %d\n\n" Lux.statelength(st)
opt = Adam(lr)
train_state = Training.TrainState(model, ps, st, opt) |> dev
@printf "[Info] Compiling Inference Model...\n"
testX, testY = (testX, testY) |> dev
# start_time = time()
# model_compiled = @compile model(testX, ps, Lux.testmode(st))
# time_to_compile = time() - start_time
best_test_loss = Inf
# @printf "[Info] Time taken to compile inference model: %0.5fs\n" time_to_compile
@printf "[Info] Starting Model Training...\n\n"
loss_fn = CrossEntropyLoss(; logits=Val(true)) |> dev
iter = 0
for epoch in 1:epochs
    for (x, y) in train_loader
        iter += 1
        start_time = time()
        _, loss, _, train_state = Training.single_train_step!(
            AutoZygote(), loss_fn, (x, y), train_state
        )
        time_taken = time() - start_time
        if iter % 100 == 0
            @printf "[Train] Epoch %3d\tIteration %6d\tLoss %.8e\tTime per \
                     Iteration %0.5f\n" epoch iter loss time_taken
        end
    end
    test_loss = loss_fn(
        first(model(testX, ps, Lux.testmode(st))), testY
    )
    @printf "[Test] Epoch %3d\tTest Loss %.8e\n" epoch test_loss
    # Generate some text here...
    eigengpt(ps)
    texts = generate_text(
        model, ps, st, seed;
        alphabet, output_length, sequence_length
    )
    for (i, (text, s)) in enumerate(zip(texts, seed))
        @printf "[Info] Seed [%d]: %s\n" i s
        @printf "[Generated Text] %s\n\n" text
    end
    if test_loss < best_test_loss
        best_test_loss = test_loss
        @printf "[Info] New best test loss: %.8e\n" best_test_loss
        @printf "[Info] Saving model...\n"
        jldsave(
            joinpath(@__DIR__, "nanogpt.jld2");
            parameters=train_state.parameters |> cdev,
            states=train_state.states |> cdev,
            alphabet=alphabet,
            model_config=model_config
        )
    end
end

anim = @animate for i ∈ 1:10:50
    histogram(eigs_gpt.layer_1.layer_2[i] |> cpu_device(), normalize = true, bins = 30, label = "eigenvalues")
    if i == 1
        x,mp = mp_plot(ps.blocks.layer_1.mlp.layer_2.weight', eigs_gpt.layer_1.layer_2[i])
    else
        x,mp = mp_plot(ps.blocks.layer_1.mlp.layer_2.weight', eigs_gpt.layer_1.layer_2[i][10-(i÷5):end])
    end
    
    plot!(x, mp, linewidth = 5, label = "Marcenko Pastur fit")
end

gif(anim, "anim1_nano.gif", fps = 1)


anim = @animate for i ∈ 1:10:50
    histogram(eigs_gpt.layer_1.layer_3[i] |> cpu_device(), normalize = true, bins = 30, label = "eigenvalues")
    if i == 1
        x,mp = mp_plot(ps.blocks.layer_1.mlp.layer_3.weight, eigs_gpt.layer_1.layer_3[i])
    else
        x,mp = mp_plot(ps.blocks.layer_1.mlp.layer_3.weight, eigs_gpt.layer_1.layer_3[i][1+(3 *(i÷5)):end])
    end
    
    plot!(x, mp, linewidth = 5, label = "Marcenko Pastur fit")
end

gif(anim, "anim2_nano.gif", fps = 1)



x,mp = mp_plot(ps.blocks.layer_1.mlp.layer_3.weight, eigs_gpt.layer_1.layer_3[1][1:end])
plot!(x, mp, linewidth = 5)
histogram!(eigs_gpt.layer_1.layer_3[5] |> cpu_device(), normalize = true, bins = 20)
histogram!(eigs_gpt.layer_1.layer_3[10] |> cpu_device(), normalize = true, bins = 20)
histogram!(eigs_gpt.layer_1.layer_3[15] |> cpu_device(), normalize = true, bins = 20)
histogram!(eigs_gpt.layer_1.layer_3[20] |> cpu_device(), normalize = true, bins = 20)
histogram!(eigs_gpt.layer_1.layer_3[25] |> cpu_device(), normalize = true, bins = 20)
histogram!(eigs_gpt.layer_1.layer_3[30] |> cpu_device(), normalize = true, bins = 20)
histogram!(eigs_gpt.layer_1.layer_3[40] |> cpu_device(), normalize = true, bins = 20)
histogram!(eigs_gpt.layer_1.layer_3[45] |> cpu_device(), normalize = true, bins = 20)
histogram(eigs_gpt.layer_1.layer_3[50] |> cpu_device(), normalize = true, bins = 20)
x,mp = mp_plot(ps.blocks.layer_1.mlp.layer_3.weight, eigs_gpt.layer_1.layer_3[50][13:end])
plot!(x, mp, linewidth = 5)

histogram(eigs_gpt.layer_6.layer_2[1] |> cpu_device(), normalize = true, bins = 20)
histogram(eigs_gpt.layer_6.layer_2[5] |> cpu_device(), normalize = true, bins = 20)
histogram(eigs_gpt.layer_6.layer_2[10] |> cpu_device(), normalize = true, bins = 20)
histogram(eigs_gpt.layer_6.layer_2[15] |> cpu_device(), normalize = true, bins = 20)
histogram(eigs_gpt.layer_6.layer_2[20] |> cpu_device(), normalize = true, bins = 20)
x,mp = mp_plot(ps.blocks.layer_6.mlp.layer_2.weight', eigs_gpt.layer_6.layer_2[20][5:end])
plot!(x, mp, linewidth = 5)
histogram(eigs_gpt.layer_6.layer_2[25] |> cpu_device(), normalize = true, bins = 20)
histogram(eigs_gpt.layer_6.layer_2[30] |> cpu_device(), normalize = true, bins = 20)
histogram(eigs_gpt.layer_6.layer_2[40] |> cpu_device(), normalize = true, bins = 20)
histogram(eigs_gpt.layer_6.layer_2[45] |> cpu_device(), normalize = true, bins = 20)
histogram(eigs_gpt.layer_6.layer_2[50] |> cpu_device(), normalize = true, bins = 20)
x,mp = mp_plot(ps.blocks.layer_6.mlp.layer_2.weight', eigs_gpt.layer_6.layer_2[50][5:end])
plot!(x, mp, linewidth = 5)

histogram(eigs_gpt.layer_6.layer_3[1] |> cpu_device(), normalize = true, bins = 20)
histogram(eigs_gpt.layer_6.layer_3[5] |> cpu_device(), normalize = true, bins = 20)
histogram(eigs_gpt.layer_6.layer_3[10] |> cpu_device(), normalize = true, bins = 20)
histogram(eigs_gpt.layer_6.layer_3[15] |> cpu_device(), normalize = true, bins = 20)
histogram(eigs_gpt.layer_6.layer_3[20] |> cpu_device(), normalize = true, bins = 20)
histogram(eigs_gpt.layer_6.layer_3[25] |> cpu_device(), normalize = true, bins = 20)
x,mp = mp_plot(ps.blocks.layer_6.mlp.layer_3.weight, eigs_gpt.layer_6.layer_3[25][30:end])
plot!(x, mp, linewidth = 5)
histogram(eigs_gpt.layer_6.layer_3[30] |> cpu_device(), normalize = true, bins = 20)
histogram(eigs_gpt.layer_6.layer_3[40] |> cpu_device(), normalize = true, bins = 20)
histogram(eigs_gpt.layer_6.layer_3[45] |> cpu_device(), normalize = true, bins = 20)
histogram(eigs_gpt.layer_6.layer_3[50] |> cpu_device(), normalize = true, bins = 20)
x,mp = mp_plot(ps.blocks.layer_6.mlp.layer_3.weight, eigs_gpt.layer_6.layer_3[50][30:end])
plot!(x, mp, linewidth = 5)

histogram(eigs_gpt.layer_2.layer_2[1] |> cpu_device(), normalize = true, bins = 20)
histogram(eigs_gpt.layer_2.layer_2[5] |> cpu_device(), normalize = true, bins = 20)
histogram(eigs_gpt.layer_2.layer_2[10] |> cpu_device(), normalize = true, bins = 20)
histogram(eigs_gpt.layer_2.layer_2[15] |> cpu_device(), normalize = true, bins = 20)
histogram(eigs_gpt.layer_2.layer_2[20] |> cpu_device(), normalize = true, bins = 20)
histogram(eigs_gpt.layer_2.layer_2[25] |> cpu_device(), normalize = true, bins = 20)
x,mp = mp_plot(ps.blocks.layer_2.mlp.layer_2.weight', eigs_gpt.layer_2.layer_2[25][2:end])
plot!(x, mp, linewidth = 5)
histogram(eigs_gpt.layer_2.layer_2[30] |> cpu_device(), normalize = true, bins = 20)
histogram(eigs_gpt.layer_2.layer_2[40] |> cpu_device(), normalize = true, bins = 20)
histogram(eigs_gpt.layer_2.layer_2[45] |> cpu_device(), normalize = true, bins = 20)
histogram(eigs_gpt.layer_2.layer_2[50] |> cpu_device(), normalize = true, bins = 20)
x,mp = mp_plot(ps.blocks.layer_2.mlp.layer_2.weight', eigs_gpt.layer_2.layer_2[50][2:end])
plot!(x, mp, linewidth = 5)

histogram(eigs_gpt.layer_5.layer_3[1] |> cpu_device(), normalize = true, bins = 20)
histogram(eigs_gpt.layer_5.layer_3[5] |> cpu_device(), normalize = true, bins = 20)
histogram(eigs_gpt.layer_5.layer_3[10] |> cpu_device(), normalize = true, bins = 20)
histogram(eigs_gpt.layer_5.layer_3[15] |> cpu_device(), normalize = true, bins = 20)
histogram(eigs_gpt.layer_5.layer_3[20] |> cpu_device(), normalize = true, bins = 20)
x,mp = mp_plot(ps.blocks.layer_5.mlp.layer_3.weight, eigs_gpt.layer_5.layer_3[20][5:end])
plot!(x, mp, linewidth = 5)
histogram(eigs_gpt.layer_5.layer_3[25] |> cpu_device(), normalize = true, bins = 20)
histogram(eigs_gpt.layer_5.layer_3[30] |> cpu_device(), normalize = true, bins = 20)
histogram(eigs_gpt.layer_5.layer_3[40] |> cpu_device(), normalize = true, bins = 20)
histogram(eigs_gpt.layer_5.layer_3[45] |> cpu_device(), normalize = true, bins = 20)
histogram(eigs_gpt.layer_5.layer_3[50] |> cpu_device(), normalize = true, bins = 20)
