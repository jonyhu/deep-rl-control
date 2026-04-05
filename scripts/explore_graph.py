import torch
import torch.nn as nn

# ── Same MLP definition from your notebook ──
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.uniform_(layer.weight, a=-0.001, b=0.001)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# ── Setup ──
value = MLP(input_dim=4, output_dim=1)

# Fake "states" — just random data, like what env would give you
state = [0.5, -0.3, 0.1, 0.8]
next_state = [0.6, -0.2, 0.15, 0.75]

state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)       # [1, 4]
next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)  # [1, 4]

# ── Two forward passes through the SAME network ──
current_value = value(state_tensor)
next_value = value(next_state_tensor)

# ═══════════════════════════════════════════════
# 1. The inputs have NO graph
# ═══════════════════════════════════════════════
print("=" * 60)
print("1. INPUTS — leaf tensors, no graph")
print("=" * 60)
print(f"  state_tensor.grad_fn       = {state_tensor.grad_fn}")
print(f"  state_tensor.requires_grad = {state_tensor.requires_grad}")
print(f"  next_state_tensor.grad_fn  = {next_state_tensor.grad_fn}")
print()

# ═══════════════════════════════════════════════
# 2. Parameters: requires_grad=True but no grad_fn (they're leaves)
# ═══════════════════════════════════════════════
print("=" * 60)
print("2. PARAMETERS — leaf tensors WITH requires_grad=True")
print("=" * 60)
print(f"  value.fc1.weight.requires_grad = {value.fc1.weight.requires_grad}")
print(f"  value.fc1.weight.grad_fn       = {value.fc1.weight.grad_fn}")
print(f"  value.fc1.weight.is_leaf       = {value.fc1.weight.is_leaf}")
print()

# ═══════════════════════════════════════════════
# 3. The OUTPUTS carry the graph
# ═══════════════════════════════════════════════
print("=" * 60)
print("3. OUTPUTS — these tensors carry the computational graph")
print("=" * 60)
print(f"  current_value.grad_fn = {current_value.grad_fn}")
print(f"  next_value.grad_fn    = {next_value.grad_fn}")
print()
print(f"  Same grad_fn object?  {current_value.grad_fn is next_value.grad_fn}")
print()

# ═══════════════════════════════════════════════
# 4. Walk each graph to see the chain of operations
# ═══════════════════════════════════════════════
def walk_graph(tensor, name, max_depth=8):
    """Walk backward through grad_fn to show the operation chain."""
    print(f"  Graph for '{name}':")
    node = tensor.grad_fn
    queue = [(node, 0)]
    visited = set()
    count = 0
    while queue and count < max_depth:
        current, level = queue.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        indent = "    " + "  | " * level
        print(f"{indent}<- {type(current).__name__}  (id: ...{id(current) % 10000:04d})")
        for child, _ in current.next_functions:
            if child is not None:
                queue.append((child, level + 1))
        count += 1
    print()

print("=" * 60)
print("4. WALKING THE GRAPHS — same structure, different objects")
print("=" * 60)
walk_graph(current_value, "current_value")
walk_graph(next_value, "next_value")

# ═══════════════════════════════════════════════
# 5. Both graphs point to the SAME parameters
# ═══════════════════════════════════════════════
print("=" * 60)
print("5. SHARED PARAMETERS — both graphs reference same weights")
print("=" * 60)

def find_leaf_params(tensor):
    """Collect all leaf parameter tensors reachable from this graph."""
    params = []
    visited = set()
    def _recurse(node):
        if node is None or id(node) in visited:
            return
        visited.add(id(node))
        for child, _ in node.next_functions:
            if child is None:
                continue
            if type(child).__name__ == "AccumulateGrad":
                params.append(child.variable)
            else:
                _recurse(child)
    _recurse(tensor.grad_fn)
    return params

current_params = find_leaf_params(current_value)
next_params = find_leaf_params(next_value)

print(f"  Params reachable from current_value: {len(current_params)} tensors")
print(f"  Params reachable from next_value:    {len(next_params)} tensors")
print()

all_same = all(
    any(cp.data_ptr() == np_.data_ptr() for np_ in next_params)
    for cp in current_params
)
print(f"  All params point to same memory? {all_same}")
print()

# ═══════════════════════════════════════════════
# 6. Backprop demo — gradients accumulate on shared params
# ═══════════════════════════════════════════════
print("=" * 60)
print("6. BACKPROP — two backward() calls, gradients accumulate")
print("=" * 60)

value.zero_grad()

loss_current = current_value.squeeze()
loss_current.backward(retain_graph=True)

grad_after_first = value.fc1.weight.grad.clone()
print(f"  fc1.weight.grad norm after 1st backward: {grad_after_first.norm():.6f}")

loss_next = next_value.squeeze()
loss_next.backward()

grad_after_second = value.fc1.weight.grad.clone()
print(f"  fc1.weight.grad norm after 2nd backward: {grad_after_second.norm():.6f}")
print(f"  Gradients accumulated (not replaced):    {(grad_after_second - grad_after_first).norm() > 0}")
