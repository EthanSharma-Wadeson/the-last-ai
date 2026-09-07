# Metrics

## Purpose

The project needs quantitative measurements so that conclusions are based on data rather than visual impressions.

## Behavioural Metrics

### Exploration Entropy

Measures how broadly the agent explores the environment.

### Location Revisit Rate

Measures how often the agent returns to previously important locations.

### Social Interaction Rate

Number of interactions per simulation window.

### Approach Probability

Probability that an agent chooses to move toward another entity.

### Avoidance Probability

Probability that an agent chooses to move away.

## Memory Metrics

### Entity Memory Strength

Magnitude of the internal representation associated with an entity.

### Memory Retrieval Frequency

How often a stored representation is accessed.

### Memory Persistence

Time required for representation strength to fall below a defined threshold.

### Memory Replacement

How often one memory is removed to make space for another.

## Prediction Metrics

### Entity Prediction Error

Difference between predicted and observed behaviour/state.

### Absence Prediction Error

Difference between expected observations of a missing entity and actual observations.

### Model Adaptation Time

Number of simulation steps required for prediction error to stabilise after environmental change.

## Representation Metrics

If entity representations are vectors, track:

- cosine similarity
- Euclidean distance
- norm
- dimensionality-reduction visualisations
- clustering

Example:

representation_before
        |
        v
disappearance
        |
        v
representation_after

Track how quickly the representation changes.

## Baseline Comparison

Every major experiment should compare against at least one baseline.

Useful baselines:

- random agent
- no-memory agent
- unlimited-memory agent
- fixed-memory agent
- non-social agent

## Statistical Analysis

Run multiple seeds.

Do not draw conclusions from one simulation.

Report:

- mean
- standard deviation
- sample count
- confidence intervals where appropriate

## Visualisation

Useful plots:

- memory strength vs time
- prediction error vs time
- behaviour before/after disappearance
- population size vs time
- representation similarity vs time
- interaction network over time