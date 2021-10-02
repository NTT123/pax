import pax


def test_mutate_new_module_list():
    a = pax.nn.Linear(3, 3)

    def add_modules(mod):
        mod.lst = []
        mod.lst.append(pax.nn.Linear(4, 4))
        return mod

    b = pax.mutate(a, with_fn=add_modules)
    assert b._name_to_kind["lst"] == pax.PaxFieldKind.MODULE
