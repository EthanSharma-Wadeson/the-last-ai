# World Specification

## Purpose

The world provides the environment in which artificial agents learn.

The first version should be deliberately small and understandable.

## World representation

Use a discrete 2D grid.

Example:

####################
#                  #
#     A            #
#          B       #
#                  #
#       C          #
#                  #
####################

Each cell may contain:

- empty space
- wall
- resource
- agent
- special object

## Time

The simulation advances in discrete ticks.

At each tick:

1. World state is observed.
2. Agents receive observations.
3. Agents update internal state.
4. Agents choose actions.
5. Actions are applied.
6. World dynamics update.
7. Learning occurs.
8. Metrics are recorded.

## Physics

Initial physics should be simple.

Possible actions:

- stay
- move north
- move south
- move east
- move west
- interact

Movement should be deterministic unless controlled noise is explicitly enabled.

## Resources

Agents may eventually require resources to maintain energy.

Resources can:

- spawn
- disappear
- be collected
- regenerate

The first experiment can omit resources entirely.

## Social environment

Agents can perceive nearby agents.

An observation should contain structured information such as:

- agent ID
- relative position
- distance
- recent interaction
- visible state

Avoid exposing information the agent could not physically observe.

## Persistence

The world should be serialisable.

A complete simulation state should be saveable and restorable.

This makes experiments reversible and reproducible.

## Procedural variation

Later versions may generate different worlds from seeds.

The seed must be stored with every experiment.

## World events

The world should support controlled events:

- agent appearance
- agent disappearance
- resource changes
- environmental changes
- communication availability
- population changes

Events should be logged.

## Disappearance

Disappearance removes an agent from the environment.

The surviving agents are not given a direct symbolic message explaining the removal unless an experiment explicitly tests that condition.

This allows researchers to measure adaptation to absence rather than response to an instruction.