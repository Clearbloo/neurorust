# NeuroRust

A neural network library written in Rust

## Roadmap

- [ ] Add opitmizers (ADAM and SGD)
- [ ] USe BLAS for matrix operations
- [ ] Graph neural networks
- [ ] Autograd

## NN

```python
model = NN()

for _e in epochs:
    output = model.forward() 
    loss_fn = loss_builder(target)
    loss = loss_fn(output)

    dL_da = loss.grad()
    for layer in model.layers:
        da_dz = layer.activation.grad()
        dz_dW = layer.output.weight_grad()

        layer.da_dW = da_dz * dz_dW
        dL_da = layer.output.activation_grad()
```
