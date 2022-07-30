Things that can be improved:

- [x] Create a core without kind
- [x] Improve how to register parameters/states.
- [x] Compute gradient with respect to trainable parameters.
- [x] How to manage random key???
- [x] Improve mixed precision and flatten module.
- [x] Improve optimizer API.
- [x] Support mixed attributes.
- [ ] Performance penalty due to tree flatten, unflatten.



- [ ] Improve impure -> pure API.

Current solution:

pax.module_and_value(net)(x, y, z)

--> new solution
pax.purecall(net, x, y, z)
Other... approach is to transformation a ... module into a pure ... init and apply.

We provide a better/general solution.

t = pax.module_value(net)(x, y)

net, t = net % (x, y)

out = pax.unsafe(net)(x, y)
net, out = pax.pure(lambda net: net, net(x))
