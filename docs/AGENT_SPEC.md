# Agent Specification

## Purpose

An agent is an artificial organism-like computational system capable of perceiving, learning, remembering, predicting, and acting within the world.

## Initial architecture

              OBSERVATION
                   |
                   v
              PERCEPTION
                   |
          +--------+--------+
          |                 |
          v                 v
       MEMORY          WORLD MODEL
          |                 |
          +--------+--------+
                   |
                   v
               MOTIVATION
                   |
                   v
              DECISION
                   |
                   v
                 ACTION
                   |
                   v
                 WORLD
                   |
                   +----> LEARNING

## State

Each agent should maintain:

- unique ID
- position
- energy
- age
- current observation
- memory
- internal representations
- motivation state
- learning parameters

## Observation

The agent receives only information permitted by its perception system.

The simulator must not accidentally provide hidden global information to the agent.

## Action selection

The initial decision system can be simple.

Possible progression:

1. random baseline
2. rule-based baseline
3. value-based learner
4. neural policy
5. learned world model
6. hybrid cognitive architecture

This progression provides useful experimental controls.

## Identity

Each agent has a unique identifier in the simulator.

However, the cognitive system should not automatically receive semantic information such as:

> "Agent 4 is your friend."

Any relationship should emerge from interaction history.

## Learning

Learning should modify internal parameters based on experience.

Every update should be reproducible where possible.

## Internal representation

The project should eventually maintain a vector or structured representation for observed entities.

For example:

entity_memory[agent_id] -> representation

The representation can encode learned information such as:

- typical location
- interaction history
- predicted behaviour
- value
- familiarity
- recency
- uncertainty

These are computational variables, not claims about emotion.

## Baseline agent

A baseline agent should have no sophisticated social memory.

This provides a control group against which the more advanced architecture can be compared.