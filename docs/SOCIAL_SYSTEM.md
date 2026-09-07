# Social System

## Purpose

The social system allows agents to develop learned relationships through interaction.

No relationship should be hard-coded as "friendship" or "attachment."

## Interaction

Agents can:

- approach
- avoid
- exchange information (structured symbolic messages — see `docs/COMMUNICATION.md`)
- share resources
- compete
- cooperate

The initial version only needs one or two interaction types.

## Relationship representation

A relationship can be represented computationally as:

relationship[A][B]

containing variables such as:

- interaction count
- cooperation rate
- conflict rate
- recency
- predicted utility
- uncertainty

## Relationship formation

A relationship emerges from repeated interactions.

Example:

A interacts with B
    ↓
A receives positive outcome
    ↓
A updates representation of B
    ↓
A becomes more likely to approach B

## Disappearance experiment

After a relationship has developed, remove one member.

The remaining agent's behaviour is measured.

Potential observations:

- movement toward previous interaction locations
- altered social exploration
- continued prediction of the missing agent
- changes in interaction preferences
- decay of entity representation

## Control condition

Remove an unfamiliar agent instead.

This establishes whether behavioural changes depend on interaction history.

## Social network

Later versions can model the population as a graph:

A ----- B
|       |
|       |
C ----- D

Edge weights represent learned interaction strength.

This allows experiments on whether information about disappeared agents persists indirectly through the surviving social network.