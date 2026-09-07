# Memory System

## Purpose

Memory is central to The Last AI.

The system should allow information about the environment and other agents to persist after the original observation is no longer available.

## Memory categories

### Short-Term Memory

Recent observations.

Properties:

- high capacity
- rapid decay
- detailed information

### Long-Term Memory

Compressed or learned representations.

Properties:

- lower capacity
- slower decay
- abstraction
- retrieval based on relevance

### Entity Memory

Information associated with particular observed entities.

Example:

Entity 12
    familiarity: 0.83
    last_seen: 18392
    expected_location: (14, 8)
    interaction_value: 0.71
    uncertainty: 0.24

## Memory decay

Memory should not simply disappear instantly.

A configurable decay function can be used.

For example:

strength(t) = strength_0 * exp(-lambda * elapsed_time)

The decay rate should be an experimental parameter.

## Retrieval

The agent should retrieve memories based on relevance rather than receiving its entire memory store every tick.

Possible retrieval signals:

- current location
- current observation
- similarity
- recency
- importance
- prediction error

## Forgetting

Forgetting is a feature, not necessarily a bug.

Experiments should compare:

- unlimited memory
- fixed-capacity memory
- decaying memory
- learned memory prioritisation

## Memory after disappearance

When an entity disappears:

- its stored representation should remain unless the memory mechanism removes it
- new observations of that entity become impossible
- the representation may decay
- predictions associated with it can be tested

This creates a measurable separation between external presence and internal representation.

## Important experiment

Compare two conditions:

### Condition A

Entity disappears after minimal interaction.

### Condition B

Entity disappears after extensive repeated interaction.

Measure whether representation persistence differs.

## Metrics

Track:

- memory strength
- retrieval frequency
- representation similarity over time
- prediction error
- behavioural bias
- memory replacement rate