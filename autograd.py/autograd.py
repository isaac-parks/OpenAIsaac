class Value:
    def __init__(self, val, children = ()):
        self.val = val
        self._backward = lambda: None
        self.grad = 0
        self.children = children

    def __repr__(self):
        return str(self.val)

    def __add__(self, newVal):
        if not isinstance(newVal, Value):
            newVal = Value(newVal)
        out = Value(self.val + newVal.val, children=(self, newVal))
        def _backward():
            self.grad += 1 + 0 * out.grad
            newVal.grad += 1 + 0 * out.grad
        out._backward = _backward

        return out

    def __mul__(self, newVal):
        if not isinstance(newVal, Value):
            newVal = Value(newVal)
        out = Value(self.val * newVal.val, children=(self, newVal))
        def _backward():
            self.grad += newVal.val * out.grad
            newVal.grad += self.val * out.grad
        out._backward = _backward

        return out

    def backward(self):
        self.grad = 1
        topo = self.build_topo()
        for node in reversed(topo):
            node._backward()


    def build_topo(self):
        topo = []
        seen = set()
        def _build(node):
            if node not in seen:
                seen.add(node)
                for child in node.children:
                    _build(child)
                topo.append(node)

        _build(self)
        return topo
