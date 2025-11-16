# Online Sequence Learning

## Explaining Output

Running this:
```python
uv run -m src.minimal
```

will give you output:
```text
After EnemyLeft: [('Pain', 0.6271488039942519), ('GoRight', 0.4150537239956475)]
After EnemyLeft->Pain: [('GoRight', 0.8106518046789989)]
After EnemyRight: [('Pain', 0.7480615054020066), ('GoLeft', 0.4150537239956475)]
After EnemyRight->Pain: [('GoLeft', 0.8106518046789989)]
```

Agent learned that after EnemyLeft comes Pain, and after both of them comes GoRight.
And it also learned EnemyRight->Pain, EnemyRight->Pain->GoLeft sequence.

The idea here is to show that the Pain concept can learn to route correctly based on where the signal came from.
Of course here we give the agent the correct answer (go left when pain is from the right side).
In practice with this setup the agent will only be able to learn EnemyLeft/Right -> Pain sequence, but that's not the point.
The point here is to give the agent the mechanism to represent the sequences of concept activations.
The correct reaction can be learned with other mechanisms, like evolutionary selection.
