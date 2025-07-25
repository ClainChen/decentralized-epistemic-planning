# decentralized-epistemic-planning

~~~bash
python entrance.py -d corridor/domain.pddl -p corridor/problem01 -ob corridor.py --strategy monte_carlo
# -d, -p, -ob is required
# --strategy is currently default by random
# use --cooperative to set the problem type to cooperative, unless it will be a neutral problem

python entrance.py -h # check the full help of this program
~~~

## Remind

- To read the code, you can start from **entrance.py**, everything start from there.
- Please read **abstract.py**
- Please put all strategy file into **policy_strategy folder**. Make sure you derive **AbstractPolicyStrategy** class, which has been defined in **abstract.py** .
- Please put all observation function file into **observation_functions fold**. Make sure you derive **AbstractObservationFunction** class, which has been defined in **abstract.py**



### Progress:

- [x] Domain parser
- [x] Problem parser
- [x] Model builder
  - [x] Model checker
    - [x] Regular checker: syntax, name, entities
    - [x] Goal conflict checker: Cooperative, Neutral
  - [x] Builder
  
- [ ] Problem Generator
- [x] Epistemic Model
- [x] Problem Solver
- [ ] Intention prediction
- [x] Virtual World generator
  - [x] Rules implementation
  - [x] All functions generator
  - [x] Filter the functions based on agent's own perspective
- [x] Epistemic Logics




## A round process:

- [x] Agent observation and update their holding functions
- [x] Agents get their successors based on their holding functions
- [x] Agents choose one of the successor that will help them to reach their goal and belief goal
- [x] check the chosen action with the ontic world functions, if it pass, then do the action, otherwise it will stay, and update their belief based on the action's condition
- [ ] Agent update their belief goals based on the observation of other agent's movement
- [x] End round



# Strategies:

- [x] Random: randomly choose a valid action

- [x] Vote Random: an extend random algorithm that will self simulate multiple times from the given model and vote the actions based on moves to goal and visit times.

  - needs to use virtual model. The simulate world is generate based on current agent's perspective.

- [x] Monte Carlo: choose the action based on Monte Carlo algorithm

  - needs to use virtual model. The simulate world is generate based on current agent's perspective.

  - [x] Monte Carlo basic algorithm

- [x] Vote Monte Carlo: an extend Monte Carlo algorithm that will self simulate multiple times from the given model and vote the actions based on moves to goal and visit times.

- [x] Greedy: decide the action based on how much the holding functions getting closer to the goal.
