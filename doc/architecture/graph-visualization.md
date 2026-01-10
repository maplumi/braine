# Graph visualization limits and scaling

The daemon can export a connectivity graph snapshot and the UI renders it as a node/edge visualization.

## UI limits

The desktop UI lets you request up to:
- **1,000 nodes**
- **50,000 edges**

These values are *request caps*: the daemon will down-select/truncate to stay within the requested bounds.

## How many edges are possible for $N$ nodes?

The theoretical maximum depends on whether you treat edges as **directed** or **undirected**:

- **Directed graph (no self-loops)**: $E_{max} = N(N-1)$
- **Undirected graph (no self-loops)**: $E_{max} = \frac{N(N-1)}{2}$

Concrete examples:
- $N=1{,}000$:
  - Directed: $999{,}000$ edges
  - Undirected: $499{,}500$ edges

With a UI cap of **50,000 edges**, a *complete* graph would be limited to roughly:
- Directed: $N \approx 224$ (since $224\cdot 223 = 49{,}952$)
- Undirected: $N \approx 317$ (since $317\cdot 316 / 2 = 50{,}086$)

In practice, Braine graphs are not complete; the daemon selects the strongest/most relevant edges first.

## Practical performance guidance

- Rendering cost scales roughly with $O(N + E)$.
- 50k edges is intentionally “stressful” for many 2D renderers; if the UI gets slow, reduce **edges** before reducing **nodes**.
- Prefer filtering by graph kind (e.g., causal vs substrate) and turn off "include isolated" unless you specifically need it.
