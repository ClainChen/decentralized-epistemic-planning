# decentralized-epistemic-planning

~~~bash
python entrance.py -d corridor/domain.pddl -p corridor/2a2i_1 -ob corridor.py --strategy complete_bfs.py --rules corridor.py --cooperative
# -d, -p, -ob is required
# --strategy now make sure you use complete_bfs.py as the strategy, that is the core strategy
# use --cooperative to set the problem type to cooperative, unless it will be a neutral problem
# use --genearte_problem to avoid simulate the model but generate all possible problems

python entrance.py -h # check the full help of this program
~~~

## Remind

- To read the code, you can start from **entrance.py**, everything start from there.
- Please read **abstract.py**
- Please put all strategy file into **policy_strategy folder**. Make sure you derive **AbstractPolicyStrategy** class, which has been defined in **abstract.py** .
- Please put all observation function file into **observation_functions fold**. Make sure you derive **AbstractObservationFunction** class, which has been defined in **abstract.py**

- **Now use complete_bfs.py as the strategy, other strategy might has some problem (random will not have any problem)**



### Progress:

- [x] Domain parser
- [x] Problem parser
- [x] Model builder
  - [x] Model checker
    - [x] Regular checker: syntax, name, entities
    - [x] Goal conflict checker: Cooperative, Neutral
  - [x] Builder
- [ ] Problem Generator
  - [x] initial world generator
  - [x] goal checker
  - [ ] parse the problem to pddl file

- [x] Epistemic Model
- [x] Problem Solver
- [x] Intention prediction
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
- [x] Agent update their belief goals based on the observation of other agent's movement
  - [ ] Fixing: it may generate a set of invalid goals, need to find a way to group the conflict goals and product them with other goals

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

- [x] BFS: generate the virtual model based on agent's own perspective, and use the centralized method to simulate the model such as BFS.
  - The only thing needs to consider to check whether the expand action is valid to the world or not.
  - The justified perspective method will be implement in this model.
  - The simulation will be slow at very start, but it will speed up later.
