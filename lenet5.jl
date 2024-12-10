using Optimization, OptimizationOptimisers, Lux
using Lux, MLUtils, Optimisers, Zygote, OneHotArrays, Random, Statistics, Printf
using Plots, LuxCUDA, LinearAlgebra, ForwardDiff
using MLDatasets: MNIST
using MLUtils: DataLoader, splitobs

CUDA.allowscalar(false)

function loadmnist(batchsize, train_split)
    # Load MNIST: Only 1500 for demonstration purposes
    # N = 30000
    dataset = MNIST(; split=:train)
    imgs = dataset.features
    labels_raw = dataset.targets

    # Process images into (H,W,C,BS) batches
    x_data = Float32.(reshape(imgs, size(imgs, 1), size(imgs, 2), 1, size(imgs, 3)))
    y_data = onehotbatch(labels_raw, 0:9)
    (x_train, y_train), (x_test, y_test) = splitobs((x_data, y_data); at=train_split)

    return (
        # Use DataLoader to automatically minibatch and shuffle the data
        DataLoader(collect.((x_train, y_train)); batchsize, shuffle=true),
        # Don't shuffle the test data
        DataLoader(collect.((x_test, y_test)); batchsize, shuffle=false)
    )
end

lux_model = Chain(
    Conv((5, 5), 1 => 20, relu),   # First conv layer (20 filters)
    MaxPool((2, 2)),               # Pooling
    Conv((5, 5), 20 => 50, relu),  # Second conv layer (50 filters)
    MaxPool((2, 2)),               # Pooling
    FlattenLayer(3),               # Flatten
    Chain(
        Dense(800 => 500, relu; init_weight = Lux.glorot_normal),   # Fully connected layer
        Dense(500 => 10, init_weight = Lux.glorot_normal)           # Output layer
    )
)

const loss = CrossEntropyLoss(; logits=Val(true))

function accuracy(model, ps_mnist, st, dataloader)
    total_correct, total = 0, 0
    st = Lux.testmode(st)
    for (x, y) in dataloader
        target_class = onecold(y) |> gpu_device()
        predicted_class = onecold(Array(first(model(x, ps_mnist, st)))) |> gpu_device()
        total_correct += sum(target_class .== predicted_class)
        total += length(target_class)
    end
    return total_correct / total
end

function convM(w)
    M = hvcat(size(w.weight,4)÷2, [w.weight[:,:,1,i] for i in 1:size(w.weight,4)]...)
    return M
end

eigs = (layer_1 = [], layer_3 = [], layer_6 = [], layer_7 = [], layer_8 = [])
rng = Xoshiro(2023)
train_dataloader, test_dataloader = loadmnist(128, 0.9) 
train_dataloader = train_dataloader .|> gpu_device()
test_dataloader = test_dataloader .|> gpu_device()
ps_mnist, st = Lux.setup(rng, lux_model)
ps_mnist = ps_mnist |> gpu_device()
st = st |> gpu_device()
train_state = Training.TrainState(lux_model, ps_mnist, st, Adam(3.0f-4))
### Warmup the model
x_proto = randn(rng, Float32, 28, 28, 1, 1) |> gpu_device()
y_proto = onehotbatch([1], 0:9) |> gpu_device()

function eigcalc(ps_mnist)
    M = convM(ps_mnist.layer_1)
    M1 = (M*M')./(size(M,2))
    push!(eigs.layer_1, svdvals(M1))
    M = convM(ps_mnist.layer_3)
    push!(eigs.layer_3, svdvals((M*M')./(size(M,2))))
    M = ps_mnist.layer_6.weight
    @show size(M)
    push!(eigs.layer_6, svdvals((M*M')./(size(M,2))))
    M = ps_mnist.layer_7.weight
    push!(eigs.layer_7, svdvals((M*M')./(size(M,2))))
    # u5,s5,v5 = svd(ps_mnist.layer_8.weight)
    # push!(eigs.layer_8, s5.*s5)
end

u5,s5,v5 = svd(ps_mnist.layer_6.weight)
# function hesseigs(ps_mnist)
#     currp = ComponentArray(layer_1 = ps_mnist.layer_1, layer_3 = ps_mnist.layer_3, layer_6_1 = ps_mnist.layer_6.layer_1, layer_6_2 = ps_mnist.layer_6.layer_2, layer_6_3 = ps_mnist.layer_6.layer_3)
#     ForwardDiff.jacobian()
    
# end

# hesseigs = []

g1 = Training.compute_gradients(AutoZygote(), loss, (x_proto, y_proto), train_state)
eigcalc(ps_mnist)

# Zygote.jacobian(x -> Zygote.gradient((args...)->loss(args...)[1], lux_model, x, st,  (x_proto|>cpu_device(), y_proto|>cpu_device()))[2], ps_mnist|>cpu_device())
# Zygote.hessian((x)->loss(lux_model, x, st, (x_proto|>cpu_device(), y_proto|>cpu_device()))[1], ps_mnist|>cpu_device())
### Lets train the model
nepochs = 30
tr_acc, te_acc = 0.0, 0.0
for epoch in 1:nepochs
    stime = time()
    for (x, y) in train_dataloader
        gs, _, _, train_state = Training.single_train_step!(
            AutoZygote(), loss, (x, y), train_state)
    end
    eigcalc(train_state.parameters)
    ttime = time() - stime
    tr_acc = accuracy(
        lux_model, train_state.parameters, train_state.states, train_dataloader) * 100
    te_acc = accuracy(
        lux_model, train_state.parameters, train_state.states, test_dataloader) * 100
    @printf "[%2d/%2d] \t Time %.2fs \t Training Accuracy: %.2f%% \t Test Accuracy: \
             %.2f%%\n" epoch nepochs ttime tr_acc te_acc
end

eigs = eigs |> cpu_device()
plt = histogram(eigs.layer_6[1])

function get_sigma(eig_max, r)
    sigma = eig_max/(1+sqrt(r))^2
    return sqrt(sigma)
end

function mp_plot(W, eigs)
    r = size(W, 1)/ size(W, 2)
    @show r
    sigma = get_sigma(maximum(eigs), r)
    @show sigma
    b = sigma^2 * (1 + sqrt(r))^2 # Largest eigenvalue
    @show b
    a = sigma^2 * (1 - sqrt(r))^2 # Smallest eigenvalue
    @show a
    x=range(a, b, size(W, 1))
    return x, (sqrt.((b .-x).*(x .-a)) ./(2*pi.*x*sigma^2*r))
end

anim = @animate for i ∈ 1:5:30
    histogram(eigs.layer_6[i], bins = 100, normalize = true, label = "eigenvalues")
    if i == 1
        x,mp = mp_plot(ps_mnist.layer_6.weight, sort(eigs.layer_6[i]))
    else
        x,mp = mp_plot(ps_mnist.layer_6.weight, sort(eigs.layer_6[i])[1:end-25-i])
    end
    plot!(x, mp, linewidth = 5, label = "Marcenko Pastur fit")
end

gif(anim, "anim_fps15.gif", fps = 1)
# x,mp = mp_plot(ps_mnist.layer_6.weight, sort(eigs.layer_6[5])[1:end-25])
# histogram(eigs.layer_6[5], bins = 100, normalize = true)
# plot!(x, mp, linewidth = 5)


# x,mp = mp_plot(ps_mnist.layer_6.weight, sort(eigs.layer_6[10])[1:end-25])
# histogram(eigs.layer_6[10], bins = 100, normalize = true)
# plot!(x, mp, linewidth = 5)

# x,mp = mp_plot(ps_mnist.layer_6.weight, sort(eigs.layer_6[15])[1:end-25])
# histogram(eigs.layer_6[15], bins = 100, normalize = true)
# plot!(x, mp, linewidth = 5)

# x,mp = mp_plot(ps_mnist.layer_6.weight, sort(eigs.layer_6[20])[1:end-25])
# histogram(eigs.layer_6[20], bins = 100, normalize = true)
# plot!(x, mp, linewidth = 5)

# b = sort(eigs.layer_6[20])[end-9:end]

# p = [train_dataloader[i][2] for i in 1:length(train_dataloader)]
# ys = hcat(p...)
# dataeigs = svdvals((ys*ys')/size(ys,2))
# train_dataloader[2][1]
# scatter(sort(dataeigs)|>cpu_device())
# scatter!(b|> cpu_device()) 