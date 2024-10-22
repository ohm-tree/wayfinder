# UCT Concurrent Implementation

(Draft)

Select Leaf:

while non terminal
- if you're non-valued, and you're not currently getting valued, i need to acquire the value lock immediately and then break. if you're currently getting valued, then i need to wait for the lock to release, at which point the node will be valued and i can continue.
- check if require new moves, then create and expand.
- This is the first critical section because this is where getnextstate and policy prior are needed.
- But once this finishes, the policy prior is fixed length again
- the thing we choose is argmax of Q+cPU. But this thing is known once we exit the critical section, so there are no problems.
- apply virtual losses always. There are no situations where we need to reject.

Get Value
- only accesses one internal state.
- We need a lock here, because other guys need to know whether a value is currently pending.

Backprop
- commutative, no locks needed.